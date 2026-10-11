#!/usr/bin/env python3
"""Second-device table: the RTX 4090 stages of one device run, speedups vs AR without expert cache."""
from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess

from specfit.experiment import canonical_json_sha256
from specfit.serial_replay import file_identity, pool_view
from run_qwen3_serial_replay import read_json, write_json
from run_specfit_main_comparison import A100_CALIBRATION, ADAPTIVE_DEPTH, ROOT, validate
from analyze_specfit_slot_capacity import metrics

BASELINE = 'ar_no_cache'
# EAGLE-3 against SpecFit with the same cache mode.
PAIRS = (('eagle_no_cache', 'specfit_adaptive_no_cache'), ('eagle_cache', 'specfit_adaptive'), ('eagle_cache', 'specfit'),
         ('specfit', 'specfit_adaptive'))
SHARED = ('protocol', 'profile', 'device', 'gpu_name', 'gpu_uuid', 'image_id', 'datasets', 'tree', 'slots_per_layer',
          'cache_policy', 'timing_scope')
SELECTS = ('specfit', 'specfit_adaptive', 'specfit_no_cache', 'specfit_adaptive_no_cache')
# Without the expert cache SpecFit selects with the cold-cache fit, with it with the natural-cache fit.
NO_CACHE_SELECTS = ('specfit_no_cache', 'specfit_adaptive_no_cache')
# Later commits of the device branch add freeze options (cost-model transfer, cold-cache fit) and change this analyzer only.
DEVICE_FILES = frozenset(('scripts/run_specfit_main_comparison.py', 'scripts/control_specfit_device.py',
                          'scripts/analyze_specfit_device.py'))
# The revised-cost stage merges the revised cost model and the route-accuracy code of exp/predictor-accuracy; the
# natural-cost selection path stays LevelSpecFit (the A100 exact-route control through it reproduced its record).
REVISED_FILES = DEVICE_FILES | frozenset(('src/specfit/main_comparison.py', 'tests/unit/test_main_comparison.py',
    'scripts/fit_specfit_revised_cost.py', 'scripts/analyze_specfit_route_accuracy.py',
    'scripts/estimate_specfit_route_accuracy.py', 'scripts/control_specfit_summarization.py',
    'scripts/analyze_specfit_summarization.py'))


def check_commit(base, commit, allowed=DEVICE_FILES):
    """A stage runs the calibration commit or a descendant that changes nothing but the device files (the revised
    stage: also the revised cost model files and documentation)."""
    if commit == base: return
    git = ['git', '-C', str(ROOT)]
    if subprocess.run([*git, 'merge-base', '--is-ancestor', base, commit]).returncode:
        raise ValueError('stage commit does not descend from the calibration commit')
    changed = set(subprocess.run([*git, 'diff', '--name-only', base, commit], capture_output=True, text=True, check=True).stdout.split())
    if allowed is REVISED_FILES: changed = {x for x in changed if not x.startswith('docs/')}
    if not changed <= allowed: raise ValueError('stage commit changes measurement code: '+' '.join(sorted(changed-allowed)))


def shared(frozen, key):
    """Identity shared by all stages; the adaptive-depth rule is part of the tree only where adaptive SpecFit runs."""
    value = frozen.get(key)
    if key != 'tree': return value
    if value.get('specfit_adaptive_depth') != (ADAPTIVE_DEPTH if 'specfit_adaptive' in frozen['reference_methods'] else None):
        raise ValueError('adaptive-depth rule differs')
    return {k: v for k, v in value.items() if k != 'specfit_adaptive_depth'}


def draft_depth(rows):
    depths = [pool_view(c['pool']).nodes[-1].depth for r in rows for c in r['cycles']]
    return sum(depths)/len(depths) if depths else None


def stage_rows(directory):
    index = read_json(directory/'result.json')
    rows = {}
    for descriptor in index['questions']:
        row = read_json(directory/descriptor['basename'], descriptor)
        rows.setdefault(row['method'], []).append(row)
    return index, rows


def stage_identity(directory, frozen, verification, index):
    return dict(directory=str(directory), git_commit=frozen['git_commit'],
        manifest=file_identity(directory/'manifest.json'), validation=file_identity(directory/'validation.json'),
        method_order=frozen['method_order'], reference_methods=frozen['reference_methods'],
        references=verification['references'], measured_requests=verification['measured_requests'],
        calibration=frozen['calibration'], cost_model_source=frozen.get('cost_model_source'),
        cost_model_mode=frozen.get('cost_model_mode'), started_ns=index.get('started_ns'))


def method_metrics(group):
    value = metrics(group)
    value['draft_depth_mean'] = draft_depth(group)
    value['committed_tokens_per_round'] = value['acceptance_lengths']['committed_length_including_root']['mean']
    return value


def pair(left, right, a, b):
    """Right over left on the same questions."""
    by_question = {r['question_id']: r for r in left}
    if sorted(by_question) != sorted(r['question_id'] for r in right): raise ValueError('paired streams cover different questions')
    faster = sum((r['output_tokens']-1)/r['decode_ns'] > (by_question[r['question_id']]['output_tokens']-1)/by_question[r['question_id']]['decode_ns']
                 for r in right)
    return dict(decode=b['decode_tps']/a['decode_tps'], request=b['request_tps']/a['request_tps'],
        h2d_per_output_token=b['h2d_per_output_token']/a['h2d_per_output_token'],
        ms_per_round=b['decode_ms_per_verification_round']/a['decode_ms_per_verification_round'],
        questions_faster=faster, questions=len(right))


def analyze(directories, calibration, output, transfers=(), revised=()):
    wrapped = read_json(calibration)
    raw = calibration.with_name(wrapped['source']['basename'])
    if file_identity(raw)['sha256'] != wrapped['source']['sha256']: raise ValueError('cost model is not its calibration record')
    fitted = wrapped['stages']['calibration']['models']['by_mode']
    report = dict(status='verified', stages={}, metrics={}, calibration=dict(cost_model=file_identity(calibration),
        combined=file_identity(raw), git_commit=wrapped['stages']['calibration']['git_commit'],
        counts=wrapped['stages']['calibration']['counts'], resources=wrapped['stages']['calibration']['resources'],
        models=wrapped['stages']['calibration']['models']), limitations=[])
    first, rows, modes, superseded = None, {}, {}, {}
    for directory in directories:
        frozen = read_json(directory/'manifest.json')
        verification = validate(directory)
        first = first or frozen
        check_commit(report['calibration']['git_commit'], frozen['git_commit'])
        for key in SHARED:
            if shared(frozen, key) != shared(first, key): raise ValueError('stages differ: '+key)
        if frozen.get('cost_model_source', 'refit_on_device') != 'refit_on_device': raise ValueError('transfer stage among the main stages')
        if 'revised_cost_model' in frozen: raise ValueError('revised-cost stage among the main stages')
        index, mine = stage_rows(directory)
        mode = frozen.get('cost_model_mode', 'natural')
        if any(m in SELECTS for m in mine) and (frozen['cost_model'] != fitted[mode]['legacy']
                or frozen['calibration']['sha256'] != report['calibration']['cost_model']['sha256']):
            raise ValueError('SpecFit stage does not use the cost model refit on this device')
        for method in [m for m in mine if m in SELECTS]:
            if method in NO_CACHE_SELECTS and mode == 'natural':
                # Selected with the natural-cache fit before the cold-cache rule: a record only, never a table row.
                superseded[method] = dict(stage=directory.name, cost_model_mode=mode, rows=mine.pop(method))
            elif (method in NO_CACHE_SELECTS) != (mode == 'cold'):
                raise ValueError('SpecFit selects with the natural-cache fit with the cache and the cold-cache fit without it')
            else:
                modes[method] = mode
        if set(mine) & set(rows): raise ValueError('a method is measured in two stages')
        rows.update(mine)
        report['stages'][directory.name] = stage_identity(directory, frozen, verification, index)
        report['limitations'] += [x for x in frozen['limitations'] if x not in report['limitations']]
    report.update(device=first['device'], gpu_name=first['gpu_name'], slots_per_layer=first['slots_per_layer'],
                  question_ids=first['datasets']['gsm8k']['question_ids'], cost_model_mode=modes)
    for method, group in rows.items():
        report['metrics'][method] = method_metrics(group)
    report['superseded'] = {}
    for method, record in superseded.items():
        value = report['superseded'][method] = dict(stage=record['stage'], cost_model_mode=record['cost_model_mode'],
                                                    metrics=method_metrics(record['rows']))
        if method in rows:
            value['cold_over_natural'] = pair(record['rows'], rows[method], value['metrics'], report['metrics'][method])
    if BASELINE in rows:
        ar = report['metrics'][BASELINE]
        report['speedup_vs_ar_no_cache'] = {m: dict(decode=v['decode_tps']/ar['decode_tps'], request=v['request_tps']/ar['request_tps'])
                                            for m, v in report['metrics'].items()}
    report['pairs'] = {}
    for left, right in PAIRS:
        if left not in rows or right not in rows: continue
        report['pairs'][f'{right}/{left}'] = pair(rows[left], rows[right], report['metrics'][left], report['metrics'][right])
    a100 = read_json(A100_CALIBRATION)['stages']['calibration']['models']['by_mode']
    for directory in transfers:
        # Same methods, questions and device as the main stages; only the cost coefficients are the A100 ones.
        frozen = read_json(directory/'manifest.json')
        verification = validate(directory)
        check_commit(report['calibration']['git_commit'], frozen['git_commit'])
        for key in SHARED:
            if shared(frozen, key) != shared(first, key): raise ValueError('transfer stage differs: '+key)
        mode = frozen.get('cost_model_mode', 'natural')
        if (frozen.get('cost_model_source') != 'a100_calibration' or frozen['cost_model'] != a100[mode]['legacy']
                or frozen['calibration']['sha256'] != file_identity(A100_CALIBRATION)['sha256']):
            raise ValueError('transfer stage does not use the recorded A100 coefficients')
        index, mine = stage_rows(directory)
        if not set(mine) <= set(modes) or any(modes[m] != mode for m in mine):
            raise ValueError('transfer methods need refit counterparts with the same cache mode')
        transfer = report.setdefault('transfer', dict(a100_cost_model={}, calibration=file_identity(A100_CALIBRATION),
                                                     stages={}, metrics={}, pairs={}))
        transfer['a100_cost_model'][mode] = a100[mode]['legacy']
        if set(mine) & set(transfer['metrics']): raise ValueError('a transfer method is measured twice')
        transfer['stages'][directory.name] = stage_identity(directory, frozen, verification, index)
        for method, group in mine.items():
            transfer['metrics'][method] = method_metrics(group)
            transfer['pairs'][f'{method} a100/refit'] = pair(rows[method], group, report['metrics'][method], transfer['metrics'][method])
        report['limitations'] += [x for x in frozen['limitations'] if x not in report['limitations']]
    for directory in revised:
        # Same method, questions, references protocol and device as the natural cached stream; selection uses the
        # revised cost model refit on that stream.
        frozen = read_json(directory/'manifest.json')
        verification = validate(directory)
        check_commit(report['calibration']['git_commit'], frozen['git_commit'], REVISED_FILES)
        for key in SHARED:
            if shared(frozen, key) != shared(first, key): raise ValueError('revised stage differs: '+key)
        if (frozen.get('cost_model_source') != 'refit_on_device' or frozen.get('cost_model_mode') != 'natural'
                or frozen['cost_model'] != fitted['natural']['legacy']
                or frozen['calibration']['sha256'] != report['calibration']['cost_model']['sha256']
                or frozen['methods'] != ['specfit_adaptive'] or modes.get('specfit_adaptive') != 'natural'):
            raise ValueError('revised stage is not cached adaptive SpecFit next to the natural refit stream')
        source = frozen['revised_cost_source']
        if (source['stage'] not in report['stages'] or source['manifest_sha256'] != canonical_json_sha256(
                read_json(Path(report['stages'][source['stage']]['directory'])/'manifest.json'))):
            raise ValueError('revised fit does not come from a main stage of this report')
        index, mine = stage_rows(directory)
        value = report.setdefault('revised', dict(stages={}, metrics={}, pairs={}))
        if 'specfit_adaptive' in value['metrics']: raise ValueError('revised stream measured twice')
        value['stages'][directory.name] = dict(stage_identity(directory, frozen, verification, index),
            revised_cost=frozen['revised_cost'], revised_cost_source=source, revised_cost_model=frozen['revised_cost_model'])
        group = mine['specfit_adaptive']
        m = value['metrics']['specfit_adaptive'] = method_metrics(group)
        value['pairs']['specfit_adaptive revised/natural'] = pair(rows['specfit_adaptive'], group,
                                                                 report['metrics']['specfit_adaptive'], m)
        if 'eagle_cache' in rows:
            value['pairs']['specfit_adaptive revised/eagle_cache'] = pair(rows['eagle_cache'], group,
                                                                         report['metrics']['eagle_cache'], m)
        if BASELINE in rows:
            ar = report['metrics'][BASELINE]
            value['speedup_vs_ar_no_cache'] = dict(decode=m['decode_tps']/ar['decode_tps'], request=m['request_tps']/ar['request_tps'])
        report['limitations'] += [x for x in frozen['limitations'] if x not in report['limitations']]
    write_json(output, report)
    for method, v in report['metrics'].items():
        s = report.get('speedup_vs_ar_no_cache', {}).get(method, {})
        print(method, v['requests'], round(v['decode_tps'], 4), round(v['request_tps'], 4), round(v['h2d_per_output_token'], 1),
              v['acceptance_lengths']['verification_rounds'], round(v['decode_ms_per_verification_round'], 1),
              round(v['selected_draft_nodes_mean'], 2) if v['acceptance_lengths']['verification_rounds'] else None,
              round(v['committed_tokens_per_round'], 3) if v['committed_tokens_per_round'] else None,
              {k: round(x, 4) for k, x in s.items()})
    for method, v in report.get('transfer', {}).get('metrics', {}).items():
        print('a100-cost', method, v['requests'], round(v['decode_tps'], 4), round(v['request_tps'], 4), round(v['h2d_per_output_token'], 1),
              v['acceptance_lengths']['verification_rounds'], round(v['decode_ms_per_verification_round'], 1),
              round(v['selected_draft_nodes_mean'], 2), round(v['committed_tokens_per_round'], 3), round(v['draft_depth_mean'], 3))
    for method, v in report.get('revised', {}).get('metrics', {}).items():
        print('revised', method, v['requests'], round(v['decode_tps'], 4), round(v['request_tps'], 4), round(v['h2d_per_output_token'], 1),
              v['acceptance_lengths']['verification_rounds'], round(v['decode_ms_per_verification_round'], 1),
              round(v['selected_draft_nodes_mean'], 2), round(v['committed_tokens_per_round'], 3), round(v['draft_depth_mean'], 3),
              {k: round(x, 4) for k, x in report['revised'].get('speedup_vs_ar_no_cache', {}).items()})
    for method, v in report['superseded'].items():
        m = v['metrics']
        print('superseded', method, v['cost_model_mode'], round(m['decode_tps'], 4), round(m['request_tps'], 4),
              {k: round(x, 4) if isinstance(x, float) else x for k, x in v.get('cold_over_natural', {}).items()})
    for key, v in (*report['pairs'].items(), *report.get('transfer', {}).get('pairs', {}).items(),
                   *report.get('revised', {}).get('pairs', {}).items()):
        print(key, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
    print('cost_model_mode', json.dumps(modes))
    for mode in ('natural', 'cold'):
        print(mode, 'legacy', json.dumps(fitted[mode]['legacy']))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__); p.add_argument('--directories', type=Path, nargs='+', required=True)
    p.add_argument('--calibration', type=Path, required=True, help='cost-model.json written by control_specfit_device.py')
    p.add_argument('--transfer', type=Path, nargs='*', default=(), help='stages selecting with the recorded A100 coefficients')
    p.add_argument('--revised', type=Path, nargs='*', default=(), help='stages selecting with the revised cost model')
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    analyze(a.directories, a.calibration, a.output, a.transfer, a.revised)
