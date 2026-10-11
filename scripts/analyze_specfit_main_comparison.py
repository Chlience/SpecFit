#!/usr/bin/env python3
"""Combine complete paired A100 repeats without averaging per-question speedups."""
from argparse import ArgumentParser
import json
from pathlib import Path

from specfit.main_comparison import summarize
from specfit.serial_replay import file_identity
from run_qwen3_serial_replay import read_json, write_json
from run_specfit_main_comparison import validate


def analyze(directories, output):
    if len(directories) != 2: raise ValueError('two complete workers required')
    workers, rows, scoring = {}, {}, {}
    for directory in directories:
        frozen = read_json(directory/'manifest.json')
        if frozen['smoke'] or frozen['worker'] in workers: raise ValueError('smoke or duplicate worker')
        verification = validate(directory)
        index = read_json(directory/'result.json')
        workers[frozen['worker']] = dict(manifest=frozen, verification=verification)
        if 'humaneval' in frozen['datasets']:
            scoring[frozen['worker']] = read_json(directory/'humaneval-score.json')
            if scoring[frozen['worker']]['status'] != 'verified': raise ValueError('HumanEval scoring incomplete')
            identity=file_identity(directory/'manifest.json')
            if (scoring[frozen['worker']]['manifest'] != {k:identity[k] for k in ('size_bytes','sha256')}
                    or scoring[frozen['worker']]['source_records'] != [d for d in index['questions'] if d['basename'].startswith('measure-humaneval-')]):
                raise ValueError('HumanEval scores do not bind these measured records')
        for descriptor in index['questions']:
            row = read_json(directory/descriptor['basename'], descriptor)
            rows.setdefault((row['dataset'], row['method']), []).append(row)
    if set(workers) != {0, 1}: raise ValueError('worker coverage changed')
    a, b = (workers[i]['manifest'] for i in (0, 1))
    for key in ('git_commit', 'image_id', 'datasets', 'cost_model', 'tree', 'cache_policy', 'methods', 'reference_methods'):
        if a[key] != b[key]: raise ValueError('worker protocols differ: '+key)
    if a.get('checkpoint_proposal') != b.get('checkpoint_proposal'):
        raise ValueError('worker checkpoint proposals differ')
    report = dict(status='verified', protocol=a['protocol'], git_commit=a['git_commit'],
                  workers=workers, datasets={}, scoring=scoring, limitations=a['limitations'])
    for dataset in a['dataset_order']:
        metrics = {m: summarize(rows[dataset, m]) for m in a['methods']}
        for method, value in metrics.items():
            value['speedup_vs_ar'] = value['decode_tps']/metrics['ar']['decode_tps']
            value['speedup_vs_cached_eagle'] = value['decode_tps']/metrics['eagle_cache']['decode_tps']
            left = {r['question_id']: r for r in rows[dataset, method] if r['repetition'] == 0}
            right = {r['question_id']: r for r in rows[dataset, method] if r['repetition'] == 1}
            value['cross_gpu_output_matches'] = sum(left[q]['output_token_ids'] == right[q]['output_token_ids'] for q in left)
            value['unique_questions'] = len(left)
            if 'checkpoint_proposal' in a:
                value['proposal_wall_ns'] = sum(c.get('proposal_wall_ns', 0)
                    for row in rows[dataset, method] for c in row['cycle_metrics'])
                value['proposal_cuda_ms'] = sum(c.get('proposal_cuda_ms', 0)
                    for row in rows[dataset, method] for c in row['cycle_metrics']) if 'chain3' in method else None
            if dataset == 'humaneval':
                value['isolated_test_passed_by_worker'] = {i: scoring[i]['summaries'][method]['passed'] for i in (0, 1)}
        report['datasets'][dataset] = metrics
    write_json(output, report)
    for name, methods in report['datasets'].items():
        print(name, {m: round(v['decode_tps'], 4) for m, v in methods.items()})


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--directories',type=Path,nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.directories,a.output)
