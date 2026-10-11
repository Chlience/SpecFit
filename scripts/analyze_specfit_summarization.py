#!/usr/bin/env python3
"""Compare AR, EAGLE with/without expert cache and SpecFit on the summarization repeat (or its pilots),
and adaptive-depth SpecFit on summarization or GSM8K. Streams of earlier sealed runs on the same GPU and questions
join with --baseline: the GSM8K baselines, the rows the complete-table run adds to the 40-article run, and EAGLE
and fixed-depth SpecFit at the slot counts of the GSM8K adaptive slot sweep."""
from argparse import ArgumentParser
from pathlib import Path

from specfit.main_comparison import NO_CACHE, reference_method, summarize, verify_record
from specfit.selection_methods import SelectionCostModel
from specfit.serial_replay import file_identity
from run_qwen3_serial_replay import read_json, write_json
from run_specfit_main_comparison import validate
from specfit_main_recovery import check_stream

PROFILES = ('summarization-specbench-v1', 'summarization-pilot-v1', 'summarization-adaptive-pilot-v1',
            'summarization-adaptive-v1', 'adaptive-gsm8k-v1', 'summarization-adaptive-complete-v1',
            'summarization-adaptive-slots-v1', 'adaptive-slot-capacity-gsm8k-v1', 'adaptive-cold0-gsm8k-v1',
            'summarization-adaptive-cold-v1', 'adaptive-cold0-gsm8k-v2')
TIMES = ('proposal_wall_ns', 'selector_wall_ns', 'target_verification_wall_ns', 'cycle_wall_ns')


def cycle_profile(rows):
    """Mean per-cycle wall ms (draft = proposal minus selection) and drafted depth where five levels were allowed."""
    cycles = [m for r in rows for m in r['cycle_metrics']]
    mean = {k[:-len('_wall_ns')]: sum(m[k] for m in cycles)/len(cycles)/1e6 for k in TIMES}
    mean['draft'] = mean['proposal'] - mean['selector']
    full = [m['actual_tree_depth'] for r in rows for m in r['cycle_metrics']
            if r['total_tokens'] - r['input_tokens'] - m['committed_output_offset'] - 1 >= 5]
    return dict(cycles=len(cycles), mean_ms=mean, full_depth_cycles=len(full),
                drafted_depth_histogram={str(d): full.count(d) for d in sorted(set(full))},
                drafted_depth_mean=sum(full)/len(full) if full else None)


def metrics(rows):
    value = summarize(rows)
    lengths = value['acceptance_lengths']
    value['h2d_per_output_token'] = value['cache_delta']['h2d_submissions']/value['output_tokens']
    value['selected_draft_nodes_mean'] = lengths['selected_draft_nodes']/lengths['verification_rounds']
    value['input_tokens'] = sum(r['input_tokens'] for r in rows)
    value['prefill_ns'] = sum(r['prefill_ns'] for r in rows)
    if reference_method(rows[0]['method']) != 'ar':
        value['cycle_profile'] = cycle_profile(rows)
    return value


def counterpart(method, base, methods):
    """The baseline (cached name base) under this method's expert-cache setting, when the comparison has it."""
    other = next(k for k, v in NO_CACHE.items() if v == base) if method in NO_CACHE else base
    return other if other in methods else None


def per_question(rows, methods, pairs):
    """Keep per-question inputs for later acceptance-versus-gain plots; ratios below still pool tokens/time."""
    def accepted(row):
        n = len(row['cycle_metrics'])
        return sum(m['accepted_draft_nodes'] for m in row['cycle_metrics'])/n if n else None
    def selected(row):
        return sum(m['selected_draft_nodes'] for m in row['cycle_metrics'])/max(1, len(row['cycle_metrics']))
    by = {m: {r['question_id']: r for r in rows[m]} for m in methods}
    per_token = lambda m, q: by[m][q]['decode_ns']/by[m][q]['output_tokens']
    trees = [m for m in methods if reference_method(m) != 'ar']
    first = by[methods[0]]
    return [dict(question_id=q, source_id=first[q].get('source_id'), input_tokens=first[q]['input_tokens'],
                 output_tokens={m: by[m][q]['output_tokens'] for m in methods},
                 decode_ns={m: by[m][q]['decode_ns'] for m in methods},
                 accepted_draft_mean={m: accepted(by[m][q]) for m in trees},
                 selected_draft_mean={m: selected(by[m][q]) for m in trees},
                 **{f'decode_vs_{name}': {m: per_token(b, q)/per_token(m, q) for m, b in pair.items()}
                    for name, pair in pairs.items()})
            for q in first]


def sealed_stream(directory, frozen, method):
    """One measured stream of a sealed dataset in an earlier run on the same GPU, rechecked against its references."""
    (dataset, spec), = frozen['datasets'].items()
    old = read_json(directory/'manifest.json')
    if (old['protocol'] != frozen['protocol'] or old['gpu_uuid'] != frozen['gpu_uuid'] or old['smoke']
            or old['datasets'].get(dataset) != spec or method not in old['methods']):
        raise ValueError('baseline run differs in protocol, GPU, dataset or methods')
    # Cached streams need the same retained slots; no-cache streams discard them before every target forward.
    retained = lambda m: m.get('retained_expert_slots', m['slots_per_layer'])
    if old['slots_per_layer'] != frozen['slots_per_layer'] or (method not in NO_CACHE and retained(old) != retained(frozen)):
        raise ValueError('baseline run differs in expert slots')
    sealed = read_json(directory/f'{dataset}-complete.json')
    model = SelectionCostModel.from_dict(old['cost_model'])
    def stream(kind, name):
        names = [f'{kind}-{dataset}-{name}-q{q}.json.gz' for q in spec['question_ids']]
        descs = [d for d in sealed['references' if kind == 'reference' else 'measurements'] if d['basename'] in names]
        if [d['basename'] for d in descs] != names:
            raise ValueError('sealed baseline coverage/order changed')
        rows = [read_json(directory/d['basename'], d) for d in descs]
        check_stream(rows, old, dataset, name, kind == 'reference')
        return rows
    rows = stream('measure', method)
    for row, reference in zip(rows, stream('reference', reference_method(method)), strict=True):
        verify_record(row, reference, model, spec['total_tokens'])
    return rows, dict(directory=str(directory), manifest=file_identity(directory/'manifest.json'),
                      git_commit=old['git_commit'], method_order=old['method_order'])


def analyze(directory, output, baselines=()):
    frozen = read_json(directory/'manifest.json')
    if frozen.get('profile') not in PROFILES or frozen['smoke']:
        raise ValueError('formal or pilot summarization or adaptive run required')
    methods = list(frozen['methods'])
    verification = validate(directory)
    index = read_json(directory/'result.json')
    rows = {}
    for descriptor in index['questions']:
        row = read_json(directory/descriptor['basename'], descriptor)
        rows.setdefault(row['method'], []).append(row)
    # Baseline streams missing here join the comparison; a stream measured again here is only a repeat check.
    sources, repeats = {m: 'this run' for m in methods}, {}
    for path, method in baselines:
        other, source = sealed_stream(path, frozen, method)
        if method in methods:
            mine, theirs = summarize(rows[method]), summarize(other)
            repeats[method] = dict(source, decode_tps=theirs['decode_tps'], request_tps=theirs['request_tps'],
                this_over_baseline_decode=mine['decode_tps']/theirs['decode_tps'],
                output_matches=sum(a['output_token_ids'] == b['output_token_ids'] for a, b in zip(rows[method], other)))
        else:
            rows[method], sources[method] = other, source
            methods.append(method)
    value = {m: metrics(rows[m]) for m in methods}
    # Tree methods compare with EAGLE and fixed-depth SpecFit under the same expert-cache setting, when present.
    trees = [m for m in methods if reference_method(m) != 'ar']
    pairs = {name: {m: b for m in trees if (b := counterpart(m, base, methods)) is not None}
             for name, base in (('eagle', 'eagle_cache'), ('specfit', 'specfit'))}
    ratio = lambda key, pair: {m: value[m][key]/value[b][key] for m, b in pair.items()}
    (name, spec), = frozen['datasets'].items()
    # Ratios use total tokens over total time; per-question speedups are not averaged.
    report = dict(status='verified', directory=str(directory), manifest=file_identity(directory/'manifest.json'),
        profile=frozen['profile'], git_commit=frozen['git_commit'], gpu_uuid=frozen['gpu_uuid'],
        method_order=frozen['method_order'], slots_per_layer=frozen['slots_per_layer'],
        retained_expert_slots=frozen.get('retained_expert_slots', frozen['slots_per_layer']),
        dataset=dict(spec, name=name), verification=verification, sources=sources, repeats=repeats, metrics=value,
        eagle_baseline=pairs['eagle'], decode_speedup_vs_eagle=ratio('decode_tps', pairs['eagle']),
        request_speedup_vs_eagle=ratio('request_tps', pairs['eagle']),
        h2d_ratio_vs_eagle=ratio('h2d_per_output_token', pairs['eagle']),
        specfit_baseline=pairs['specfit'], decode_speedup_vs_fixed_specfit=ratio('decode_tps', pairs['specfit']),
        per_question=per_question(rows, methods, pairs), limitations=frozen['limitations'])
    if 'eagle_cache' in value and 'specfit' in value:
        # Keys of the earlier summarization reports: every method against cached EAGLE and cached SpecFit.
        e, s = value['eagle_cache'], value['specfit']
        report.update(decode_speedup_specfit_vs_cached_eagle=s['decode_tps']/e['decode_tps'],
            request_speedup_specfit_vs_cached_eagle=s['request_tps']/e['request_tps'],
            h2d_ratio_specfit_vs_cached_eagle=s['h2d_per_output_token']/e['h2d_per_output_token'],
            decode_speedup_vs_cached_eagle={m: value[m]['decode_tps']/e['decode_tps'] for m in methods},
            decode_speedup_vs_specfit={m: value[m]['decode_tps']/s['decode_tps'] for m in methods})
        if 'eagle_no_cache' in value:
            report['decode_speedup_specfit_vs_uncached_eagle'] = s['decode_tps']/value['eagle_no_cache']['decode_tps']
    # The paper's baseline is AR without expert cache; the original summarization profile only has cached AR.
    ar = next((m for m in ('ar_no_cache', 'ar') if m in value), None)
    if ar is not None:
        report.update(ar_baseline=ar, decode_speedup_vs_ar={m: value[m]['decode_tps']/value[ar]['decode_tps'] for m in methods},
            request_speedup_vs_ar={m: value[m]['request_tps']/value[ar]['request_tps'] for m in methods})
    write_json(output, report)
    nan = float('nan')
    print('slots', report['slots_per_layer'], 'retained', report['retained_expert_slots'], 'ar_baseline', ar)
    print('method decode_tps vs_ar request_tps h2d/token accepted_draft_mean selected_nodes_mean truncated source')
    for m in methods:
        v = value[m]
        print(m, round(v['decode_tps'], 3), round(report.get('decode_speedup_vs_ar', {}).get(m, nan), 3), round(v['request_tps'], 3),
              round(v['h2d_per_output_token'], 2), round(v['acceptance_lengths']['accepted_draft_length']['mean'], 3),
              round(v['selected_draft_nodes_mean'], 2), v['truncated'], 'this run' if sources[m] == 'this run' else sources[m]['directory'])
    print('method eagle vs_eagle request_vs_eagle h2d_vs_eagle specfit vs_specfit draft_ms selector_ms verify_ms cycle_ms drafted_depth_mean histogram')
    for m in trees:
        c, t = value[m]['cycle_profile'], value[m]['cycle_profile']['mean_ms']
        print(m, pairs['eagle'].get(m), round(report['decode_speedup_vs_eagle'].get(m, nan), 4),
              round(report['request_speedup_vs_eagle'].get(m, nan), 4), round(report['h2d_ratio_vs_eagle'].get(m, nan), 4),
              pairs['specfit'].get(m), round(report['decode_speedup_vs_fixed_specfit'].get(m, nan), 4),
              *(round(t[k], 2) for k in ('draft', 'selector', 'target_verification', 'cycle')),
              round(c['drafted_depth_mean'], 3) if c['drafted_depth_mean'] is not None else None, c['drafted_depth_histogram'])
    for m, r in repeats.items():
        print('repeat', m, r['directory'], round(r['decode_tps'], 3), round(r['this_over_baseline_decode'], 4), r['output_matches'])


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--baseline', nargs=2, action='append', default=[], metavar=('DIRECTORY', 'METHOD'),
                   help='measured stream of a sealed dataset from an earlier run on the same GPU and questions')
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    analyze(a.directory, a.output, [(Path(d), m) for d, m in a.baseline])
