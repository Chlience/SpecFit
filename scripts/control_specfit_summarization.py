#!/usr/bin/env python3
"""Host controller: Spec-Bench summarization pilot, smoke or one formal repeat on one GPU; with --adaptive, also the
GSM8K adaptive-depth run (stage gsm8k, SpecFit references reused from --reference-from) and complements of the
adaptive paper runs that reuse all references of the sealed run given by --reference-from: summarization stages
complete (AR with cache, EAGLE and adaptive SpecFit without cache) and slotsN (EAGLE vs adaptive SpecFit with N
retained expert slots), and GSM8K stages gsm8k-slotsN (adaptive SpecFit only; 0 = no cache). Stage
gsm8k-cold-slots0 is a fresh zero-slot GSM8K run of adaptive SpecFit with the cold-cache cost model, and stage
complete-cold a fresh run of adaptive SpecFit without cache on the 40 summarization articles with that model.
Stages gsm8k-routeNN are fresh 48-slot runs of adaptive SpecFit on the even GSM8K question IDs that select with
simulated expert-predictor routes of NN% per-expert accuracy; gsm8k-route100 predicts every expert correctly and
is the matched control, run through the same code path from the same cold start. Stage gsm8k-revised is that control
with the revised cost model given by --revised-cost.

Only the chosen GPU must be free; the other A100 may run an independent job on its own socket. Experts are
registered mmap pages of the packed cache, and page cache is shared by every container reading the same file,
so a job beside another one needs its own copy (--packed-cache) whose page cache sits on its GPU's NUMA node."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_specfit_main_comparison import DATA, GSM, PACKED, UUIDS, checked, docker, snapshot, write

PREP = '/data/artifacts/evaluation-workloads/20260925/prepared'
LIMITS = dict(pilot=2*3600+300, smoke=2*3600+300, formal=14*3600+300, gsm8k=4*3600+300, complete=6*3600+300,
              slots12=6*3600+300, slots24=6*3600+300, slots36=6*3600+300)
LIMITS.update({'gsm8k-slots'+n: 3*3600+300 for n in ('0', '12', '24', '36')}, **{'gsm8k-cold-slots0': 4*3600+300, 'complete-cold': 4*3600+300})
LIMITS.update({'gsm8k-route'+n: 3*3600+300 for n in ('100', '90', '80', '70', '60', '50')}, **{'gsm8k-revised': 3*3600+300})
# Stages of one invocation share one --reference-from; formal, run beside gsm8k, reuses nothing.
FRESH = {s for s in LIMITS if 'route' in s or 'revised' in s}
GROUPS = ({'gsm8k', 'formal'}, {'complete', 'slots12', 'slots24', 'slots36'}, {'gsm8k-slots0', 'gsm8k-slots12', 'gsm8k-slots24', 'gsm8k-slots36'},
          {'gsm8k-cold-slots0'}, {'complete-cold'}, FRESH)
REUSE = {s for s in LIMITS if s not in FRESH} - {'pilot', 'smoke', 'formal', 'gsm8k-cold-slots0', 'complete-cold'}


def main(args):
    root, source, GPU = args.directory.resolve(), args.source.resolve(), args.gpu
    root.mkdir(parents=True, exist_ok=True)
    relative = '/data/'+str(root.relative_to(DATA))
    began = time.monotonic(); active = {}
    log = (root/'controller-events.jsonl').open('x')
    samples = (root/'resource-samples.jsonl').open('x')
    def event(**fields):
        log.write(json.dumps(dict(time_ns=time.time_ns(), **fields))+'\n'); log.flush()
    def cleanup():
        for name, (process, stream, stage) in active.items():
            # Exact names assigned by this controller; never signal foreign jobs.
            subprocess.run(['docker', 'stop', '--time', '20', name], capture_output=True, timeout=40)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.terminate()
            stream.close()
        active.clear()
    def common(stage):
        return ['--directory', f'{relative}/{stage}-gpu{GPU}', '--dataset', GSM, '--prepared-root', PREP,
                '--git-commit', args.git_commit, '--worker', str(GPU)]
    def freeze(stage):
        cmd = docker(source)
        # Keep reference checks off both workers' cores (12-35) and memory nodes; fresh route-accuracy
        # freezes stay on socket 0, clear of the reference checks of a run on the other GPU.
        cpus, mems = ('6-11', '0') if stage in FRESH else ('40-47', '3')
        cmd = cmd[:3]+['--cpuset-cpus', cpus, '--cpuset-mems', mems]+cmd[3:]
        cmd += ['python', 'scripts/run_specfit_main_comparison.py', 'freeze', *common(stage)]
        if not stage.startswith('gsm8k'): cmd += ['--summarization']
        if stage in ('smoke', 'pilot'): cmd += ['--'+stage]
        if args.adaptive: cmd += ['--adaptive']
        if stage == 'complete': cmd += ['--complete-table']
        if 'slots' in stage: cmd += ['--slot-capacity', stage.rsplit('slots', 1)[1]]
        if 'cold' in stage: cmd += ['--cold-cost-model']
        if 'route' in stage: cmd += ['--route-accuracy', str(int(stage.rsplit('route', 1)[1])/100)]
        if stage == 'gsm8k-revised':
            cmd += ['--route-accuracy', '1.0', '--revised-cost', args.revised_cost]
        if stage in REUSE: cmd += ['--reference-from', '/data/'+str(args.reference_from.resolve().relative_to(DATA))]
        if args.resume_from is not None and stage == 'formal':
            previous = '/data/'+str(args.resume_from.resolve().relative_to(DATA))
            cmd += ['--resume-from', f'{previous}/formal-gpu{GPU}']
        result = checked(cmd, timeout=7200 if args.resume_from is not None or stage in REUSE else 600)
        (root/f'freeze-{stage}-gpu{GPU}.log').write_text(result.stdout+result.stderr)
    def launch(stage):
        name = f'specfit-summ-{root.name}-{stage}-{GPU}'
        limit = LIMITS[stage] if args.max_seconds is None else min(LIMITS[stage], args.max_seconds)
        cmd = docker(source, GPU, name, args.packed_cache)+['timeout', '--signal=TERM', '--kill-after=20s', str(limit)+'s',
            'python', '-u', 'scripts/run_specfit_main_comparison.py', 'run', *common(stage)]
        stream = (root/f'{stage}-gpu{GPU}.log').open('x')
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        process.stdin.close(); active[name] = (process, stream, stage)
        event(phase='launched', stage=stage, gpu=GPU, name=name, pid=process.pid, command=cmd)
    def monitor():
        while active:
            sample = snapshot(); samples.write(json.dumps(sample)+'\n'); samples.flush()
            for name, (process, stream, stage) in list(active.items()):
                rc = process.poll()
                if rc is not None:
                    stream.close(); del active[name]
                    event(phase='worker_exited', stage=stage, gpu=GPU, returncode=rc)
                    if rc: raise RuntimeError(f'{stage} GPU {GPU} exited {rc}')
                    continue
                running = occupied(sample)
                if running:
                    top = subprocess.run(['docker', 'top', name, '-eo', 'pid'], capture_output=True, text=True, timeout=15)
                    if top.returncode and process.poll() is not None:
                        continue
                    if top.returncode: raise RuntimeError('cannot verify GPU process ownership: '+top.stderr)
                    own = {int(line.strip()) for line in top.stdout.splitlines() if line.strip().isdigit()}
                    foreign = [line for line in running if int(line.split(',')[1]) not in own]
                    if foreign: raise RuntimeError('external GPU process observed: '+repr(foreign))
            if active: time.sleep(5)
    def occupied(sample):
        return [line for line in sample['apps'].splitlines() if line.split(',')[0].strip() == UUIDS[GPU]]
    try:
        initial = snapshot()
        if occupied(initial): raise RuntimeError('GPU already occupied before launch')
        write(root/'controller-start.json', dict(commit=args.git_commit, pid=os.getpid(), gpus=[GPU], stages=args.stages,
            prepared_root=PREP, resume_from=None if args.resume_from is None else str(args.resume_from),
            max_seconds=args.max_seconds, packed_cache=str(args.packed_cache), adaptive=args.adaptive,
            reference_from=None if args.reference_from is None else str(args.reference_from), initial=initial))
        for stage in args.stages:
            freeze(stage)
            if occupied(snapshot()): raise RuntimeError('GPU occupied during preparation')
            launch(stage)
            monitor()
            event(phase='worker_verified', stage=stage, gpu=GPU)
        write(root/'controller-result.json', dict(status='complete', elapsed_seconds=time.monotonic()-began, final=snapshot()))
    except BaseException as error:
        cleanup()
        write(root/'controller-failure.json', dict(status='failed', error=repr(error), elapsed_seconds=time.monotonic()-began, final=snapshot()))
        raise
    finally:
        cleanup(); log.close(); samples.close()


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--git-commit', required=True)
    p.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    p.add_argument('--stages', nargs='+', choices=tuple(LIMITS), default=['smoke', 'formal'])
    p.add_argument('--max-seconds', type=int, help='cap each worker below its stage limit')
    p.add_argument('--packed-cache', type=Path, default=Path(PACKED), help='host directory mounted as /packed-cache')
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--adaptive', action='store_true',
                   help='adaptive-depth SpecFit: pilot alone, or gsm8k and/or formal (40 summarization articles)')
    p.add_argument('--revised-cost', help='gsm8k-revised: fitted revised cost model, relative to the source tree')
    p.add_argument('--reference-from', type=Path,
                   help='sealed run whose references are reused: 48-slot GSM8K run with SpecFit references (gsm8k), '
                        'the adaptive summarization run (complete, slotsN) or the adaptive GSM8K run (gsm8k-slotsN)')
    args = p.parse_args()
    if args.adaptive and args.stages != ['pilot'] and not any(set(args.stages) <= group for group in GROUPS):
        p.error('--adaptive runs the pilot alone, the gsm8k and formal stages, or complements of one adaptive run')
    reuse = [s for s in args.stages if s in REUSE]
    if bool(reuse) != (args.reference_from is not None) or (reuse and not args.adaptive):
        p.error('stages that reuse references are adaptive runs and need --reference-from')
    if ('gsm8k-revised' in args.stages) != (args.revised_cost is not None) or (args.revised_cost and not args.adaptive):
        p.error('stage gsm8k-revised is adaptive and needs --revised-cost')
    main(args)
