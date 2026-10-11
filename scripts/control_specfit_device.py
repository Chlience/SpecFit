#!/usr/bin/env python3
"""Host controller on rtx4090-host: second-device GSM8K runs on its RTX 4090 24GB.

A comparison stage freezes --methods with its own references, --slots expert slots per layer and, for SpecFit,
the cost model refit on this GPU (--cost-model; the recorded A100 calibration instead gives the transfer ablation),
whose cold-cache fit SpecFit without cache selects with (--cold-cost-model, a stage of no-cache SpecFit methods).
--revised-cost adds the revised cost model fitted on this GPU's natural-fit stream (a stage of cached adaptive SpecFit).
--calibrate instead runs the frozen calibration protocol of the cost model on this GPU (freeze, one worker per batch,
combine, validate) and writes that refit as cost-model.json.
The 4090 is GPU 1 on NUMA node 1 (CPUs 26-51); the workers and the freeze containers stay on that node, and the
packed cache was written there so its page cache is node-local. Only the 4090 must be free: GPU 0 belongs to other
users. Any foreign process on the 4090 aborts the stage."""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_specfit_main_comparison import IMAGE, checked, snapshot, write

# Same identity as specfit.main_comparison.DEVICES, which the host Python (no torch) cannot import.
DEVICES = {'rtx4090': 'GPU-00000000-0000-0000-0000-000000000003'}
DATA = Path.home()/'specfit'
GSM = '/data/workloads/gsm8k/e383d21cf8c005b0fcba4b8d27a2800a075fed0d/math_reasoning.jsonl'
GPU, WORKER_CPUS, FREEZE_CPUS, NODE = 1, '30-41', '42-49', '1'
CALIBRATE = ['python', '-u', 'scripts/benchmark_qwen3_selection_refinement.py']


def docker(source, name=None):
    cmd = ['docker', 'run', '--rm', '--init']
    if name:
        cmd += ['--name', name, '--label', 'specfit-controller='+name, '--gpus', f'device={GPU}',
                '--cpuset-cpus', WORKER_CPUS, '--cpuset-mems', NODE, '--shm-size', '8g',
                '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864']
    else:
        cmd += ['--cpuset-cpus', FREEZE_CPUS, '--cpuset-mems', NODE]
    return cmd+['--env', 'PYTHONPATH=/workspace/specfit/src', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'SPECFIT_PACKED_CACHE_ROOT=/packed-cache', '--volume', f'{DATA}:/data', '--volume', f'{source}:/workspace/specfit',
        '--volume', f'{DATA}/model_zoo:/models:ro', '--volume', f'{DATA}/model-cache:/packed-cache:ro',
        '--workdir', '/workspace/specfit', IMAGE]


def main(args):
    root, source, uuid = args.directory.resolve(), args.source.resolve(), DEVICES[args.device]
    root.mkdir(parents=True, exist_ok=True)
    relative = '/data/'+str(root.relative_to(DATA))
    began, active = time.monotonic(), {}
    log = (root/f'controller-events-{args.stage}.jsonl').open('x')
    samples = (root/f'resource-samples-{args.stage}.jsonl').open('x')
    def event(**fields):
        log.write(json.dumps(dict(time_ns=time.time_ns(), **fields))+'\n'); log.flush()
    def cleanup():
        for name, (process, stream) in active.items():
            # Exact names assigned by this controller; never signal foreign jobs.
            subprocess.run(['docker', 'stop', '--time', '20', name], capture_output=True, timeout=40)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.terminate()
            stream.close()
        active.clear()
    def occupied(sample):
        return [line for line in sample['apps'].splitlines() if line.split(',')[0].strip() == uuid]
    def prepare(label, cmd, seconds=600):
        result = checked(docker(source)+cmd, timeout=seconds)
        (root/f'{label}.log').write_text(result.stdout+result.stderr)
        if occupied(snapshot()): raise RuntimeError('GPU occupied during preparation')
    def work(label, cmd, seconds):
        """One worker container on the 4090, watched until it exits; any foreign process on the GPU aborts."""
        name = f'specfit-device-{root.name}-{label}'
        cmd = docker(source, name)+['timeout', '--signal=TERM', '--kill-after=20s', f'{seconds}s', *cmd]
        stream = (root/f'{label}.log').open('x')
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        process.stdin.close(); active[name] = (process, stream)
        event(phase='launched', stage=label, name=name, pid=process.pid, command=cmd)
        while active:
            sample = snapshot(); samples.write(json.dumps(sample)+'\n'); samples.flush()
            rc = process.poll()
            if rc is not None:
                stream.close(); active.clear()
                event(phase='worker_exited', stage=label, returncode=rc)
                if rc: raise RuntimeError(f'{label} exited {rc}')
                break
            running = occupied(sample)
            if running:
                top = subprocess.run(['docker', 'top', name, '-eo', 'pid'], capture_output=True, text=True, timeout=15)
                if top.returncode and process.poll() is not None:
                    continue
                if top.returncode: raise RuntimeError('cannot verify GPU process ownership: '+top.stderr)
                own = {int(line.strip()) for line in top.stdout.splitlines() if line.strip().isdigit()}
                foreign = [line for line in running if int(line.split(',')[1]) not in own]
                if foreign: raise RuntimeError('external GPU process observed: '+repr(foreign))
            time.sleep(5)
        event(phase='worker_verified', stage=label)
    try:
        initial = snapshot()
        if occupied(initial): raise RuntimeError('GPU already occupied before launch')
        write(root/f'controller-start-{args.stage}.json', dict(commit=args.git_commit, pid=os.getpid(), device=args.device,
            gpu=GPU, uuid=uuid, stage=args.stage, calibrate=args.calibrate, methods=args.methods, slots=args.slots,
            cost_model=None if args.cost_model is None else str(args.cost_model), cold_cost_model=args.cold_cost_model,
            revised_cost=None if args.revised_cost is None else str(args.revised_cost),
            max_seconds=args.max_seconds,
            worker_cpus=WORKER_CPUS, freeze_cpus=FREEZE_CPUS, node=NODE, initial=initial))
        if args.calibrate:
            calibrate(args, root/args.stage, f'{relative}/{args.stage}', prepare, work)
        else:
            common = ['--directory', f'{relative}/{args.stage}', '--dataset', GSM, '--git-commit', args.git_commit]
            cmd = ['python', 'scripts/run_specfit_main_comparison.py', 'freeze', *common, '--device', args.device,
                   '--methods', *args.methods, '--device-slots', str(args.slots)]
            if args.cost_model is not None:
                cmd += ['--cost-model', '/data/'+str(args.cost_model.resolve().relative_to(DATA))]
            if args.cold_cost_model:
                cmd += ['--cold-cost-model']
            if args.revised_cost is not None:
                cmd += ['--revised-cost', '/data/'+str(args.revised_cost.resolve().relative_to(DATA))]
            prepare(f'freeze-{args.stage}', cmd)
            work(args.stage, ['python', '-u', 'scripts/run_specfit_main_comparison.py', 'run', *common], args.max_seconds)
        write(root/f'controller-result-{args.stage}.json', dict(status='complete', elapsed_seconds=time.monotonic()-began,
                                                                final=snapshot()))
    except BaseException as error:
        cleanup()
        write(root/f'controller-failure-{args.stage}.json', dict(status='failed', error=repr(error),
                                                                 elapsed_seconds=time.monotonic()-began, final=snapshot()))
        raise
    finally:
        cleanup(); log.close(); samples.close()


def calibrate(args, directory, inside, prepare, work):
    """Frozen calibration protocol at this device's slots; the combined fit becomes the stage's cost model."""
    directory.mkdir()
    manifest = f'{inside}/manifest.json'
    prepare('calibration-freeze', [*CALIBRATE, 'freeze', '--stage', 'calibration', '--output', manifest, '--dataset', GSM])
    batches = [f'{inside}/calibration-batch-{b}.json'
               for b in range(len(json.loads((directory/'manifest.json').read_text())['batches']))]
    for b, output in enumerate(batches):
        # The entry enforces the frozen 90-minute budget itself; the container limit only backs it up.
        work(f'calibration-batch-{b}', [*CALIBRATE, 'run', '--manifest', manifest, '--batch', str(b), '--output', output,
                                        '--dataset', GSM, '--device', args.device], 5400+600)
    combined = f'{inside}/calibration.json'
    # Combining revalidates every batch; each file is then validated once more on its own.
    prepare('calibration-combine', [*CALIBRATE, 'combine', '--manifest', manifest, '--batch-results', *batches,
                                    '--output', combined, '--dataset', GSM], 3600)
    for output in (*batches, combined):
        prepare('calibration-validate-'+Path(output).stem, [*CALIBRATE, 'validate', '--manifest', manifest,
                                                             '--output', output, '--dataset', GSM], 3600)
    raw = (directory/'calibration.json').read_bytes()
    # The runner reads stages.calibration.models, the layout of the A100 calibration record.
    write(directory/'cost-model.json', dict(artifact_type='device_cost_model', device=args.device,
        source=dict(basename='calibration.json', sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)),
        stages=dict(calibration=json.loads(raw))))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--git-commit', required=True)
    p.add_argument('--device', choices=tuple(DEVICES), default='rtx4090')
    p.add_argument('--stage', required=True, help='run name inside --directory')
    p.add_argument('--calibrate', action='store_true', help='refit the cost model on this GPU instead of a comparison')
    p.add_argument('--methods', nargs='+')
    p.add_argument('--slots', type=int)
    p.add_argument('--cost-model', type=Path, help='cost model under ~/specfit for SpecFit methods: '
                   'this GPU\'s refit, or the recorded A100 calibration (transfer ablation)')
    p.add_argument('--cold-cost-model', action='store_true', help='select with the cold-cache fit of --cost-model')
    p.add_argument('--revised-cost', type=Path, help='revised cost model under ~/specfit fitted on this GPU\'s natural-fit '
                   'stream (fit_specfit_revised_cost.py --device); cached adaptive SpecFit selects with it')
    p.add_argument('--max-seconds', type=int, default=12*3600+300)
    parsed = p.parse_args()
    if parsed.calibrate and (parsed.methods or parsed.slots is not None or parsed.cost_model or parsed.cold_cost_model):
        p.error('--calibrate takes no methods, slots or cost model')
    if parsed.cold_cost_model and parsed.cost_model is None:
        p.error('--cold-cost-model selects with the cold-cache fit of --cost-model')
    if parsed.revised_cost is not None and (parsed.cost_model is None or parsed.cold_cost_model
                                            or parsed.methods != ['specfit_adaptive']):
        p.error('--revised-cost is a stage of cached adaptive SpecFit next to the natural fit of --cost-model')
    if not parsed.calibrate and (not parsed.methods or parsed.slots is None):
        p.error('a comparison needs --methods and --slots')
    main(parsed)
