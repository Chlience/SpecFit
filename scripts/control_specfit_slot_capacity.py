#!/usr/bin/env python3
"""Host controller: run the GSM8K slot sweep one A100 worker at a time (GPU 0 by default); other GPUs stay idle."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_specfit_main_comparison import DATA, GSM, PREP, UUIDS, checked, docker, snapshot, write

REFERENCES = DATA/'artifacts/specfit-main-comparison/20260924-r2'


def main(args):
    root, source = args.directory.resolve(), args.source.resolve()
    root.mkdir(parents=True, exist_ok=True)
    relative = '/data/'+str(root.relative_to(DATA))
    references = '/data/'+str(REFERENCES.relative_to(DATA))
    began = time.monotonic(); active = {}
    log = (root/'controller-events.jsonl').open('x')
    samples = (root/'resource-samples.jsonl').open('x')
    def event(**fields):
        log.write(json.dumps(dict(time_ns=time.time_ns(), **fields))+'\n'); log.flush()
    def cleanup():
        for name, (process, stream, gpu, stage) in active.items():
            # Exact names assigned by this controller; never signal foreign jobs.
            subprocess.run(['docker', 'stop', '--time', '20', name], capture_output=True, timeout=40)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.terminate()
            stream.close()
        active.clear()
    def common(stage, gpu):
        return ['--directory', f'{relative}/{stage}-gpu{gpu}', '--dataset', GSM, '--prepared-root', PREP,
                '--git-commit', args.git_commit, '--worker', str(gpu)]
    def freeze(stage, gpu, slots):
        cmd = docker(source)+['python', 'scripts/run_specfit_main_comparison.py', 'freeze', *common(stage, gpu),
            '--slot-capacity', str(slots), '--reference-from', f'{references}/formal-gpu{gpu}']
        result = checked(cmd, timeout=1800)
        (root/f'freeze-{stage}-gpu{gpu}.log').write_text(result.stdout+result.stderr)
    def launch(stage, gpu):
        name = f'specfit-slots-{root.name}-{stage}-{gpu}'
        cmd = docker(source, gpu, name)+['timeout', '--signal=TERM', '--kill-after=20s', str(4*3600+300)+'s',
            'python', '-u', 'scripts/run_specfit_main_comparison.py', 'run', *common(stage, gpu)]
        stream = (root/f'{stage}-gpu{gpu}.log').open('x')
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        process.stdin.close(); active[name] = (process, stream, gpu, stage)
        event(phase='launched', stage=stage, gpu=gpu, name=name, pid=process.pid, command=cmd)
    def monitor():
        while active:
            sample = snapshot(); samples.write(json.dumps(sample)+'\n'); samples.flush()
            for name, (process, stream, gpu, stage) in list(active.items()):
                rc = process.poll()
                if rc is not None:
                    stream.close(); del active[name]
                    event(phase='worker_exited', stage=stage, gpu=gpu, returncode=rc)
                    if rc: raise RuntimeError(f'{stage} GPU {gpu} exited {rc}')
                    continue
                running = [line for line in sample['apps'].splitlines() if line.split(',')[0].strip() == UUIDS[gpu]]
                if running:
                    top = subprocess.run(['docker', 'top', name, '-eo', 'pid'], capture_output=True, text=True, timeout=15)
                    if top.returncode and process.poll() is not None:
                        continue
                    if top.returncode: raise RuntimeError('cannot verify GPU process ownership: '+top.stderr)
                    own = {int(line.strip()) for line in top.stdout.splitlines() if line.strip().isdigit()}
                    foreign = [line for line in running if int(line.split(',')[1]) not in own]
                    if foreign: raise RuntimeError('external GPU process observed: '+repr(foreign))
            if active: time.sleep(5)
    try:
        initial = snapshot()
        if initial['apps'].strip(): raise RuntimeError('GPU already occupied before launch')
        write(root/'controller-start.json', dict(commit=args.git_commit, pid=os.getpid(), slots=args.slots, gpus=args.gpus, schedule='sequential_workers',
                                                 references=str(REFERENCES), initial=initial))
        for slots in args.slots:
            stage = f'slots{slots}'
            for gpu in args.gpus:
                # Sequential workers avoid host memory/PCIe contention between the two offloaded runs.
                freeze(stage, gpu, slots)
                if snapshot()['apps'].strip(): raise RuntimeError('GPU occupied during preparation')
                launch(stage, gpu)
                monitor()
                event(phase='worker_verified', slots=slots, gpu=gpu)
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
    p.add_argument('--slots', type=int, nargs='+', default=[16, 48, 32, 24])
    p.add_argument('--gpus', type=int, nargs='+', choices=(0, 1), default=[0])
    main(p.parse_args())
