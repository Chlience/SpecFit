#!/usr/bin/env python3
"""Compare SpecFit with EAGLE at each retained expert-slot capacity on GPU 0 (0 = no expert cache)."""
from argparse import ArgumentParser
import json
from pathlib import Path

from specfit.main_comparison import summarize
from specfit.serial_replay import file_identity
from run_qwen3_serial_replay import read_json, write_json
from run_specfit_main_comparison import validate

CACHED = ('eagle_cache', 'specfit')
# Zero retained slots: every method discards completed experts before every target forward.
EXPECTED = {0: ('ar_no_cache', 'eagle_no_cache', 'specfit_no_cache')}
PROTOCOL_KEYS = ('protocol', 'image_id', 'datasets', 'cost_model', 'tree', 'profile')


def metrics(rows):
    value = summarize(rows)
    lengths = value['acceptance_lengths']
    value['h2d_per_output_token'] = value['cache_delta']['h2d_submissions']/value['output_tokens']
    value['selected_draft_nodes_mean'] = lengths['selected_draft_nodes']/lengths['verification_rounds']
    value['decode_ms_per_verification_round'] = value['decode_ns']/1e6/lengths['verification_rounds']
    return value


def capacity(roots, slots, gpus):
    methods = EXPECTED.get(slots, CACHED)
    eagle, specfit = methods[-2:]
    workers, rows, frozen = {}, {}, {}
    for gpu in gpus:
        found = [r/f'slots{slots}-gpu{gpu}' for r in roots if (r/f'slots{slots}-gpu{gpu}/manifest.json').is_file()]
        if len(found) != 1: raise ValueError(f'expected one run for {slots} slots on GPU {gpu}')
        directory = found[0]
        frozen[gpu] = read_json(directory/'manifest.json')
        retained = frozen[gpu].get('retained_expert_slots', frozen[gpu]['slots_per_layer'])
        if frozen[gpu]['worker'] != gpu or retained != slots or list(frozen[gpu]['methods']) != list(methods):
            raise ValueError('slot worker identity changed')
        if validate(directory)['status'] != 'verified': raise ValueError('slot worker not verified')
        index = read_json(directory/'result.json')
        mine = {}
        for descriptor in index['questions']:
            row = read_json(directory/descriptor['basename'], descriptor)
            mine.setdefault(row['method'], []).append(row)
            rows.setdefault(row['method'], []).append(row)
        value = {m: metrics(mine[m]) for m in methods}
        workers[gpu] = dict(directory=str(directory), manifest=file_identity(directory/'manifest.json'), method_order=frozen[gpu]['method_order'],
            gpu_uuid=frozen[gpu]['gpu_uuid'], reference_source=frozen[gpu]['recovery']['source_manifest'], metrics=value,
            decode_speedup_specfit_vs_eagle=value[specfit]['decode_tps']/value[eagle]['decode_tps'])
    first = frozen[gpus[0]]
    for other in frozen.values():
        for key in ('git_commit', 'cache_policy', 'slots_per_layer', *PROTOCOL_KEYS):
            if other[key] != first[key]: raise ValueError('worker protocols differ: '+key)
    pooled = {m: metrics(rows[m]) for m in methods}
    if len(gpus) == 2:
        for method in methods:
            left = {r['question_id']: r for r in rows[method] if r['repetition'] == 0}
            right = {r['question_id']: r for r in rows[method] if r['repetition'] == 1}
            pooled[method]['cross_gpu_output_matches'] = sum(left[q]['output_token_ids'] == right[q]['output_token_ids'] for q in left)
    e, s = pooled[eagle], pooled[specfit]
    return first, dict(retained_expert_slots=slots, working_slots_per_layer=first['slots_per_layer'], methods=list(methods),
        git_commit=first['git_commit'], cache_policy=first['cache_policy'], gpus=list(gpus), workers=workers, metrics=pooled,
        decode_speedup_specfit_vs_eagle=s['decode_tps']/e['decode_tps'],
        request_speedup_specfit_vs_eagle=s['request_tps']/e['request_tps'],
        h2d_ratio_specfit_vs_eagle=s['h2d_per_output_token']/e['h2d_per_output_token'])


def concurrency_check(stopped, report):
    """Cached-EAGLE decode speed of a stopped concurrent run vs the sequential run on the same questions."""
    frozen = read_json(stopped/'manifest.json')
    value = report['capacities'].get(frozen.get('retained_expert_slots', frozen['slots_per_layer']))
    if value is None or frozen['worker'] not in value['workers']: raise ValueError('no sequential run to compare')
    sequential = Path(value['workers'][frozen['worker']]['directory'])
    pairs = []
    for path in sorted(stopped.glob('measure-gsm8k-eagle_cache-q*.json.gz')):
        left, right = read_json(path), read_json(sequential/path.name)
        if left['output_token_ids'] != right['output_token_ids']: raise ValueError('outputs differ: '+path.name)
        pairs.append((left, right))
    tps = lambda rows: sum(r['output_tokens']-1 for r in rows)*1e9/sum(r['decode_ns'] for r in rows)
    concurrent, alone = tps([a for a, _ in pairs]), tps([b for _, b in pairs])
    return dict(directory=str(stopped), manifest=file_identity(stopped/'manifest.json'), slots_per_layer=frozen['slots_per_layer'],
        gpu=frozen['worker'], method='eagle_cache', questions=len(pairs), concurrent_decode_tps=concurrent,
        sequential_decode_tps=alone, sequential_over_concurrent=alone/concurrent)


def analyze(roots, slots, gpus, baseline, output, stopped=None):
    report = dict(status='verified', remote_directories=[str(r) for r in roots], capacities={}, limitations=[])
    first = None
    for n in sorted(slots):
        frozen, report['capacities'][n] = capacity(roots, n, gpus)
        # Capacities may come from different source commits; the measured protocol must not change.
        first = first or frozen
        for key in PROTOCOL_KEYS:
            if frozen[key] != first[key]: raise ValueError('capacities differ: '+key)
        report['limitations'] += [x for x in frozen['limitations'] if x not in report['limitations']]
    if 0 in report['capacities']:
        # The paper's speedup baseline: AR without expert cache, measured in the zero-slot run.
        ar = report['capacities'][0]['metrics']['ar_no_cache']
        report['baseline'] = dict(method='ar_no_cache', decode_tps=ar['decode_tps'], request_tps=ar['request_tps'])
        for value in report['capacities'].values():
            value['speedup_vs_ar_no_cache'] = {m: dict(decode=v['decode_tps']/ar['decode_tps'], request=v['request_tps']/ar['request_tps'])
                                               for m, v in value['metrics'].items()}
    base = read_json(baseline)
    def brief(group):
        value = {m: dict(decode_tps=v['decode_tps'], request_tps=v['request_tps']) for m, v in group.items()}
        value['decode_speedup_specfit_vs_cached_eagle'] = group['specfit']['decode_tps']/group['eagle_cache']['decode_tps']
        return value
    # r2 ran both GPUs concurrently; kept as a reference, not merged into the sequential sweep.
    report['concurrent_r2_48'] = dict(source=file_identity(baseline), source_git_commit=base['source_git_commit'],
        pooled=brief(base['metrics']), workers={g: brief(base['workers'][str(g)]['metrics']) for g in gpus})
    if stopped: report['concurrency_check'] = concurrency_check(stopped, report)
    write_json(output, report)
    for n, value in report['capacities'].items():
        e, s = (value['metrics'][m] for m in value['methods'][-2:])
        print(n, value['methods'], round(e['decode_tps'], 3), round(s['decode_tps'], 3),
              round(value['decode_speedup_specfit_vs_eagle'], 4), round(value['request_speedup_specfit_vs_eagle'], 4),
              round(e['h2d_per_output_token'], 1), round(s['h2d_per_output_token'], 1), round(s['selected_draft_nodes_mean'], 2))
        if 'speedup_vs_ar_no_cache' in value:
            print('  vs ar_no_cache', {m: round(v['decode'], 4) for m, v in value['speedup_vs_ar_no_cache'].items()})
    for g, value in report['concurrent_r2_48']['workers'].items():
        print('r2-48-gpu'+str(g), {m: round(v['decode_tps'], 3) for m, v in value.items() if isinstance(v, dict)},
              round(value['decode_speedup_specfit_vs_cached_eagle'], 4))
    if stopped:
        c = report['concurrency_check']
        print('concurrency', c['questions'], round(c['concurrent_decode_tps'], 3), round(c['sequential_decode_tps'], 3),
              round(c['sequential_over_concurrent'], 4))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__); p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--slots', type=int, nargs='+', default=[0, 12, 16, 24])
    p.add_argument('--gpus', type=int, nargs='+', choices=(0, 1), default=[0])
    p.add_argument('--baseline', type=Path, default=Path('docs/results/specfit-main-comparison-r2-gsm8k.json'))
    p.add_argument('--concurrency-check', type=Path, help='stopped concurrent worker directory at a measured capacity')
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    analyze(a.roots, a.slots, a.gpus, a.baseline, a.output, a.concurrency_check)
