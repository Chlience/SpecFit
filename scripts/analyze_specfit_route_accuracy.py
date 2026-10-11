#!/usr/bin/env python3
"""Compare adaptive SpecFit selecting with simulated predictor routes (90%, 80%) against exact routes, cached EAGLE and
AR without cache on the 40 even GSM8K question IDs of GPU 0. EAGLE and AR are the sealed full-length streams of the
paper table, restricted to the same questions. A route-accuracy run at p = 1.0 selects with the true routes through the
same code path and cold start and is the matched control; the exact-route stream of the paper table, restricted the
same way, is kept for the record. A revised-cost run is that control selecting with the revised cost model and is
compared with it."""
from argparse import ArgumentParser
import json
from pathlib import Path

from specfit.main_comparison import (NO_CACHE, PROTOCOL, RevisedCostModel, predicted_routes, reference_method,
    summarize, verify_record)
from specfit.selection_methods import SelectionCostModel
from run_qwen3_serial_replay import read_json, write_json
from run_specfit_main_comparison import validate

A = Path('/data/artifacts')
EXACT = A/'specfit-adaptive/20260926-r1/gsm8k-gpu0', 'specfit_adaptive'
EAGLE = A/'specfit-main-comparison/20260924-r2/formal-gpu0', 'eagle_cache'
AR = A/'specfit-slot-capacity/20260925-r5/slots0-gpu0', 'ar_no_cache'


def stream(directory, method, questions, gpu_uuid):
    """Rows of a sealed GSM8K stream on the same GPU for the given questions, rechecked against their references."""
    frozen = read_json(directory/'manifest.json')
    retained = frozen.get('retained_expert_slots', frozen['slots_per_layer'])
    if (frozen['protocol'] != PROTOCOL or frozen['gpu_uuid'] != gpu_uuid or frozen['smoke'] or method not in frozen['methods']
            or frozen['datasets']['gsm8k']['total_tokens'] != 512 or (method not in NO_CACHE and retained != 48)):
        raise ValueError(f'{directory} differs in protocol, GPU, dataset, method or retained slots')
    sealed = read_json(directory/'gsm8k-complete.json')
    descriptors = {d['basename']: d for kind in ('references', 'measurements') for d in sealed[kind]}
    model = SelectionCostModel.from_dict(frozen['cost_model'])
    revised = frozen.get('revised_cost_model')
    revised = None if revised is None else RevisedCostModel.from_dict(revised)
    rows = []
    for q in questions:
        name, source = f'measure-gsm8k-{method}-q{q}.json.gz', f'reference-gsm8k-{reference_method(method)}-q{q}.json.gz'
        row = read_json(directory/name, descriptors[name])
        verify_record(row, read_json(directory/source, descriptors[source]), model, 512, frozen.get('route_accuracy'),
                      revised)
        rows.append(row)
    return rows


def metrics(rows):
    value = summarize(rows)
    lengths = value['acceptance_lengths']
    rounds = lengths['verification_rounds']
    value.update(h2d_per_output_token=value['cache_delta']['h2d_submissions']/value['output_tokens'],
        selected_draft_nodes_mean=lengths['selected_draft_nodes']/rounds,
        accepted_draft_length_mean=lengths['accepted_draft_length']['mean'],
        committed_length_mean=lengths['committed_length_including_root']['mean'],
        decode_ms_per_verification_round=value['decode_ns']/1e6/rounds)
    depths = [max(n['depth'] for n in c['pool']['candidates']) for r in rows for c in r['cycles']]
    if depths:
        value.update(draft_depth_mean=sum(depths)/len(depths), stop_before_level5_share=sum(d < 5 for d in depths)/len(depths))
    return value


def realized(directory, accuracy):
    """Fraction of true routed experts the simulated predictor kept, over every node and layer the reference saw."""
    index, hits, slots = read_json(directory/'result.json'), 0, 0
    for descriptor in index['references']:
        row = read_json(directory/descriptor['basename'], descriptor)
        for i, cycle in enumerate(row['cycles']):
            guess = predicted_routes(cycle['atlas'], accuracy, (row['dataset'], row['question_id'], i))
            hits += sum(len(set(t) & set(g)) for tl, gl in zip(cycle['atlas'], guess) for t, g in zip(tl, gl))
            slots += sum(len(t) for tl in cycle['atlas'] for t in tl)
    return hits/slots


def main(args):
    runs, checks = {}, {}
    for directory in args.runs:
        frozen, report = read_json(directory/'manifest.json'), validate(directory)
        revised = frozen['profile'] == 'revised-cost-gsm8k-v1'
        if ((frozen['profile'] != 'route-accuracy-gsm8k-v1' and not revised) or frozen['worker'] != 0
                or report['status'] != 'verified' or (revised and frozen['route_accuracy'] != 1.0)):
            raise ValueError(f'{directory} is not a verified GPU 0 route-accuracy or revised-cost run')
        label = 'revised' if revised else str(frozen['route_accuracy'])
        if label in runs:
            raise ValueError(f'two runs for {label}')
        runs[label] = directory, frozen
        # validate() matched every measured output and per-forward logits/route digest to the run's own reference.
        checks[label] = dict(directory=str(directory), git_commit=frozen['git_commit'], route_seed=frozen['route_seed'],
            references=report['references'], measured_requests=report['measured_requests'],
            manifest_sha256=report['manifest_sha256'], **(dict(revised_cost=frozen['revised_cost']) if revised else {}))
    questions = {tuple(f['datasets']['gsm8k']['question_ids']) for _, f in runs.values()}
    if len(questions) != 1:
        raise ValueError('route-accuracy runs cover different questions')
    questions = list(questions.pop())
    gpu = {f['gpu_uuid'] for _, f in runs.values()}.pop()
    base = {name: metrics(stream(*source, questions, gpu)) for name, source in (('exact', EXACT), ('eagle_cache', EAGLE), ('ar', AR))}
    table = [dict(route_accuracy=1.0, cost_model='natural', realized_accuracy=1.0, origin='paper-table stream (record)',
                  **base['exact'])]
    order = sorted(runs.items(), key=lambda item: (item[0] == 'revised', -item[1][1]['route_accuracy']))
    for label, (directory, frozen) in order:
        accuracy = frozen['route_accuracy']
        table.append(dict(route_accuracy=accuracy, cost_model='revised' if label == 'revised' else 'natural',
                          realized_accuracy=realized(directory, accuracy), origin=str(directory),
                          **metrics(stream(directory, 'specfit_adaptive', questions, gpu))))
    matched = next((row for row in table[1:] if row['route_accuracy'] == 1.0 and row['cost_model'] == 'natural'), None)
    if 'revised' in runs and matched is None:
        raise ValueError('the revised-cost run needs its matched exact-route control')
    for row in table:
        for name in ('ar', 'eagle_cache'):
            row[f'decode_speedup_vs_{name}'] = row['decode_tps']/base[name]['decode_tps']
            row[f'request_speedup_vs_{name}'] = row['request_tps']/base[name]['request_tps']
        row['decode_vs_exact_routes'] = row['decode_tps']/base['exact']['decode_tps']
        if matched is not None:
            row['decode_vs_matched_control'] = row['decode_tps']/matched['decode_tps']
            row['request_vs_matched_control'] = row['request_tps']/matched['request_tps']
    report = dict(status='verified', questions=questions, runs=checks,
        matched_control=None if matched is None else matched['origin'],
        route_accuracy_definition='per routed expert of every node and layer: kept with probability p, otherwise '
                                  'replaced by a uniformly drawn expert outside the true top-8; the root is included',
        baselines=dict(exact=f'{EXACT[0]} {EXACT[1]}', eagle_cache=f'{EAGLE[0]} {EAGLE[1]}', ar=f'{AR[0]} {AR[1]}',
                       note='taken from the sealed paper-table streams, not re-measured in the route-accuracy runs'),
        eagle_cache=base['eagle_cache'], ar=base['ar'], table=table,
        limitations=[('EAGLE and AR are' if matched is not None else 'Exact routes, EAGLE and AR are')
                     + ' the first-pass streams of 80 questions restricted to the 40 even question IDs; each '
                     'route-accuracy stream is cold-started on question 402, so expert-cache history differs slightly.',
                     'The simulated predictor replaces routed experts uniformly at random outside the true top-8; real '
                     'predictor errors are correlated with the true routes.']
                    + (['The revised cost model is fitted on the odd question IDs of the paper-table exact-route '
                        'stream; its run and the matched control are single streams on the even IDs, so the paired '
                        'difference includes drift and acceptance sampling.'] if 'revised' in runs else []))
    if args.output is not None:
        print(json.dumps(write_json(args.output, report)))
    labels = [('record 1.00', table[0]),
              *(('revised' if r['cost_model'] == 'revised' else f"{r['route_accuracy']:.2f}", r) for r in table[1:])]
    for name, row in (*labels, ('eagle', base['eagle_cache']), ('ar', base['ar'])):
        print(f"{name} tps {row['decode_tps']:.3f}/{row['request_tps']:.3f} h2d/tok {row['h2d_per_output_token']:.1f} "
              f"rounds {row['acceptance_lengths']['verification_rounds']} ms/round {row['decode_ms_per_verification_round']:.1f} "
              f"nodes {row['selected_draft_nodes_mean']:.2f} accepted {row['accepted_draft_length_mean']:.3f} "
              f"committed {row['committed_length_mean']:.3f} correct {row['numeric_correct']}", end='')
        if 'draft_depth_mean' in row:
            print(f" depth {row['draft_depth_mean']:.2f} stop<5 {row['stop_before_level5_share']:.3f}", end='')
        if 'realized_accuracy' in row:
            print(f" realized {row['realized_accuracy']:.4f} vsAR {row['decode_speedup_vs_ar']:.3f} "
                  f"vsEAGLE {row['decode_speedup_vs_eagle_cache']:.4f} vsExact {row['decode_vs_exact_routes']:.4f}", end='')
            if 'decode_vs_matched_control' in row:
                print(f" vsMatched {row['decode_vs_matched_control']:.4f}", end='')
        print()


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', type=Path)
    main(p.parse_args())
