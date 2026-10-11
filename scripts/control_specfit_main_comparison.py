#!/usr/bin/env python3
"""Host controller: gate the two resident workers on a complete model smoke test."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import time

IMAGE='sha256:64f6847e922677a269c82e23f140cca5ceafd5899e17002d52bc05035eec6567'
UUIDS=('GPU-00000000-0000-0000-0000-000000000001','GPU-00000000-0000-0000-0000-000000000002')
DATA=Path('/path/to/data')
GSM='/data/workloads/gsm8k/e383d21cf8c005b0fcba4b8d27a2800a075fed0d/math_reasoning.jsonl'
PREP='/data/artifacts/evaluation-workloads/20260922/prepared'
PACKED='/path/to/packed-cache'


def checked(args, **kw):
    return subprocess.run(args,check=True,capture_output=True,text=True,**kw)


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def docker(source, gpu=None, name=None, packed=PACKED):
    cmd=['docker','run','--rm','--init']
    if name:cmd += ['--name',name,'--label','specfit-controller='+name]
    if gpu is not None:
        cmd += ['--gpus','device='+str(gpu),'--cpuset-cpus',('12-23','24-35')[gpu],
                '--cpuset-mems',str(gpu+1),'--shm-size','8g','--ulimit','memlock=-1','--ulimit','stack=67108864']
    cmd += ['--env','PYTHONPATH=/workspace/specfit/src','--env','PYTHONDONTWRITEBYTECODE=1',
        '--env','HF_ENDPOINT=https://huggingface.co','--env','SPECFIT_PACKED_CACHE_ROOT=/packed-cache',
        '--volume',str(DATA)+':/data','--volume',str(source)+':/workspace/specfit',
        '--volume','/path/to/model_zoo:/models:ro',
        '--volume',str(packed)+':/packed-cache:ro',
        '--workdir','/workspace/specfit',IMAGE]
    return cmd


def snapshot():
    apps=checked(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],timeout=15).stdout
    gpus=checked(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,power.draw','--format=csv,noheader'],timeout=15).stdout
    return dict(time_ns=time.time_ns(),apps=apps,gpus=gpus)


def main(args):
    root, source=args.directory.resolve(),args.source.resolve()
    root.mkdir(parents=True,exist_ok=True)
    relative='/data/'+str(root.relative_to(DATA))
    began=time.monotonic();active={}
    log=(root/'controller-events.jsonl').open('x')
    samples=(root/'resource-samples.jsonl').open('x')
    def event(**fields):
        log.write(json.dumps(dict(time_ns=time.time_ns(),**fields))+'\n');log.flush()
    def cleanup():
        for name,(process,stream,gpu,stage) in active.items():
            # Exact names assigned by this controller; never signal foreign jobs.
            subprocess.run(['docker','stop','--time','20',name],capture_output=True,timeout=40)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.terminate()
            stream.close()
        active.clear()
    def freeze(stage,gpu):
        cmd=docker(source)+['python','scripts/run_specfit_main_comparison.py','freeze',
            '--directory',f'{relative}/{stage}-gpu{gpu}','--dataset',GSM,'--prepared-root',PREP,
            '--git-commit',args.git_commit,'--worker',str(gpu)]
        if stage=='smoke':cmd+=['--smoke']
        if args.checkpoint_default_comparison:cmd+=['--checkpoint-default-comparison']
        if args.resume_from is not None and stage=='formal':
            previous='/data/'+str(args.resume_from.resolve().relative_to(DATA))
            cmd+=['--resume-from',f'{previous}/{stage}-gpu{gpu}']
        result=checked(cmd,timeout=7200 if args.resume_from is not None else 120)
        (root/f'freeze-{stage}-gpu{gpu}.log').write_text(result.stdout+result.stderr)
    def launch(stage,gpu):
        name=f'specfit-main-{root.name}-{stage}-{gpu}'
        limit=2*3600+300 if args.checkpoint_default_comparison else (4*3600+300 if stage=='smoke' else 72*3600+300)
        cmd=docker(source,gpu,name)+['timeout','--signal=TERM','--kill-after=20s',str(limit)+'s',
            'python','-u','scripts/run_specfit_main_comparison.py','run',
            '--directory',f'{relative}/{stage}-gpu{gpu}','--dataset',GSM,'--prepared-root',PREP,
            '--git-commit',args.git_commit,'--worker',str(gpu)]
        stream=(root/f'{stage}-gpu{gpu}.log').open('x')
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        process.stdin.close();active[name]=(process,stream,gpu,stage)
        event(phase='launched',stage=stage,gpu=gpu,name=name,pid=process.pid,command=cmd)
    def monitor():
        while active:
            sample=snapshot();samples.write(json.dumps(sample)+'\n');samples.flush()
            for name,(process,stream,gpu,stage) in list(active.items()):
                rc=process.poll()
                if rc is not None:
                    stream.close();del active[name]
                    event(phase='worker_exited',stage=stage,gpu=gpu,returncode=rc)
                    if rc:raise RuntimeError(f'{stage} GPU {gpu} exited {rc}')
                    continue
                running=[line for line in sample['apps'].splitlines() if line.split(',')[0].strip()==UUIDS[gpu]]
                if running:
                    top=subprocess.run(['docker','top',name,'-eo','pid'],capture_output=True,text=True,timeout=15)
                    if top.returncode and process.poll() is not None:
                        continue
                    if top.returncode:raise RuntimeError('cannot verify GPU process ownership: '+top.stderr)
                    own={int(line.strip()) for line in top.stdout.splitlines() if line.strip().isdigit()}
                    foreign=[line for line in running if int(line.split(',')[1]) not in own]
                    if foreign:raise RuntimeError('external GPU process observed: '+repr(foreign))
            if active:time.sleep(5)
    def grade(stage,gpu):
        cmd=['python3',str(source/'scripts/score_specfit_humaneval.py'),
            '--directory',str(root/f'{stage}-gpu{gpu}'),'--prepared',str(DATA/'artifacts/evaluation-workloads/20260922/prepared/humaneval/6d43fb980f9fee3c892a914eda09951f772ad10d/requests.jsonl'),
            '--output',str(root/f'{stage}-gpu{gpu}/humaneval-score.json')]
        r=checked(cmd,timeout=11000)
        (root/f'scoring-{stage}-gpu{gpu}.log').write_text(r.stdout+r.stderr)
    try:
        initial=snapshot()
        if initial['apps'].strip():raise RuntimeError('GPU already occupied before launch')
        write(root/'controller-start.json',dict(commit=args.git_commit,pid=os.getpid(),initial=initial))
        if args.checkpoint_default_comparison:
            if args.resume_from is not None:raise ValueError('small comparison cannot resume the main experiment')
            event(phase='small_comparison',reference_gate='all four method references checked before measurement')
        elif args.resume_from is None:
            freeze('smoke',0);launch('smoke',0);monitor();grade('smoke',0)
            event(phase='smoke_passed')
        else:
            previous=args.resume_from.resolve()
            validation=json.loads((previous/'smoke-gpu0/validation.json').read_text())
            scores=json.loads((previous/'smoke-gpu0/humaneval-score.json').read_text())
            events=[json.loads(line) for line in (previous/'controller-events.jsonl').read_text().splitlines()]
            if (validation['status']!='verified' or scores['status']!='verified'
                    or not any(e['phase']=='smoke_passed' for e in events)):
                raise RuntimeError('previous full-model smoke did not pass')
            event(phase='prior_smoke_reused',directory=str(previous))
        for gpu in (0,1):freeze('formal',gpu)
        if snapshot()['apps'].strip():raise RuntimeError('GPU occupied during preparation')
        for gpu in (0,1):launch('formal',gpu)
        monitor()
        if not args.checkpoint_default_comparison:
            for gpu in (0,1):grade('formal',gpu)
        result=checked(docker(source)+['python','scripts/analyze_specfit_main_comparison.py',
            '--directories',relative+'/formal-gpu0',relative+'/formal-gpu1','--output',relative+'/summary.json'],timeout=7200)
        (root/'analysis.log').write_text(result.stdout+result.stderr)
        write(root/'controller-result.json',dict(status='complete',elapsed_seconds=time.monotonic()-began,final=snapshot()))
    except BaseException as error:
        cleanup()
        write(root/'controller-failure.json',dict(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-began,final=snapshot()))
        raise
    finally:
        cleanup();log.close();samples.close()


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--git-commit',required=True)
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--checkpoint-default-comparison',action='store_true')
    main(p.parse_args())
