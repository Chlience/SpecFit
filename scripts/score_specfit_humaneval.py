#!/usr/bin/env python3
"""Grade archived HumanEval responses in an isolated, unprivileged CPU container."""
from argparse import ArgumentParser
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

IMAGE = 'sha256:64f6847e922677a269c82e23f140cca5ceafd5899e17002d52bc05035eec6567'

# This process runs only in the isolated container. Each program has a fresh
# interpreter, a fixed wall/CPU budget, bounded output, and no host mounts.
CHILD = r'''
import contextlib,io,json,resource,signal,sys
p=json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
signal.alarm(8)
status='passed';error=None
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  scope={}
  exec(compile(p['program'],'<generated>','exec'),scope)
  exec(compile(p['test']+'\ncheck('+p['entry_point']+')','<hidden-tests>','exec'),scope)
except BaseException as e:
 status='failed';error=repr(e)[:500]
print(json.dumps(dict(nonce=p['nonce'],status=status,error=error)))
'''
CONTAINER = r'''
import json,subprocess,sys,uuid
payload=json.loads(sys.stdin.read());child=payload['child'];results=[]
for row in payload['jobs']:
 p=dict(row,nonce=uuid.uuid4().hex)
 try:
  r=subprocess.run([sys.executable,'-I','-c',child],input=json.dumps(p),capture_output=True,text=True,timeout=10)
  value=json.loads(r.stdout) if r.returncode==0 else dict(status='failed',error='child exit '+str(r.returncode))
  if r.returncode==0 and value.pop('nonce',None)!=p['nonce']:raise ValueError('child output identity')
 except subprocess.TimeoutExpired:value=dict(status='timeout',error='10 second wall limit')
 except Exception as e:value=dict(status='failed',error=repr(e)[:500])
 results.append(dict(key=row['key'],**value))
print(json.dumps(dict(results=results)))
'''


def digest(path):
    with path.open('rb') as stream:
        value = hashlib.sha256(stream.read()).hexdigest()
    return dict(size_bytes=path.stat().st_size, sha256=value)


def read(path, descriptor=None):
    if descriptor and digest(path) != {k: descriptor[k] for k in ('size_bytes', 'sha256')}:
        raise ValueError('record identity changed')
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def score(directory, prepared, output):
    manifest, index = read(directory/'manifest.json'), read(directory/'result.json')
    if index['status'] != 'complete': raise ValueError('generation incomplete')
    expected = manifest['datasets']['humaneval']
    if digest(prepared) != {k: expected['identity'][k] for k in ('size_bytes', 'sha256')}:
        raise ValueError('hidden-test provenance changed')
    inputs = {r['question_id']: r for r in map(json.loads, prepared.read_text().splitlines())}
    jobs, descriptors, extraction_failures = [], [], []
    for q in expected['question_ids']:
        truth = inputs[q]['evaluation']
        jobs.append(dict(key=f'canonical/{q}', program=truth['original_prompt']+truth['canonical_solution'],
                         test=truth['test'], entry_point=truth['entry_point']))
    for d in index['questions']:
        if not d['basename'].startswith('measure-humaneval-'): continue
        row = read(directory/d['basename'], d)
        descriptors.append(d)
        key = f"{row['method']}/{row['question_id']}"
        truth = inputs[row['question_id']]['evaluation']
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', row['output_text'], flags=re.DOTALL)
        if len(blocks) > 1:
            extraction_failures.append(dict(key=key, status='failed', error='multiple code blocks'))
            continue
        program = blocks[0].strip() if blocks else row['output_text'].strip()
        jobs.append(dict(key=key, program=program, test=truth['test'], entry_point=truth['entry_point']))
    command = ['docker','run','--rm','-i','--network=none','--read-only','--user','65534:65534',
        '--cap-drop=ALL','--security-opt','no-new-privileges','--pids-limit','64','--memory','512m','--cpus','1',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=64m,mode=1777', '--env','OMP_NUM_THREADS=1',
        '--env','OPENBLAS_NUM_THREADS=1', '--entrypoint','python', IMAGE,'-I','-c',CONTAINER]
    result = subprocess.run(command,input=json.dumps(dict(child=CHILD,jobs=jobs)),text=True,capture_output=True,
                            timeout=len(jobs)*12+120,check=True)
    rows = json.loads(result.stdout)['results']
    canonical = [r for r in rows if r['key'].startswith('canonical/')]
    if len(canonical) != len(expected['question_ids']) or any(r['status'] != 'passed' for r in canonical):
        raise ValueError('isolated canonical self-check failed: '+repr([r for r in canonical if r['status'] != 'passed']))
    rows = [r for r in rows if not r['key'].startswith('canonical/')]+extraction_failures
    methods = manifest['methods']
    if {r['key'] for r in rows} != {f'{m}/{q}' for m in methods for q in expected['question_ids']} or len(rows) != len(methods)*len(expected['question_ids']):
        raise ValueError('scoring coverage changed')
    report = dict(status='verified', protocol='specfit-humaneval-isolated-v1',image_id=IMAGE,
        manifest=digest(directory/'manifest.json'),source_records=descriptors,prepared=digest(prepared),
        canonical_self_checks=len(canonical), scoring_script=digest(Path(__file__)), results=rows,
        summaries={m:dict(questions=len(expected['question_ids']),passed=sum(r['status']=='passed' for r in rows if r['key'].startswith(m+'/'))) for m in methods},
        limits=dict(cpu_seconds=5,wall_seconds=10,container_memory_mb=512),
        semantics='One greedy sample per question; full returned code or one code block; no repair. Adapted Instruct prompts, not original-completion leaderboard.')
    with output.open('x') as stream: json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(report['summaries']),flush=True)


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();score(a.directory,a.prepared,a.output)
