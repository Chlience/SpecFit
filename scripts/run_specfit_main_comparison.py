#!/usr/bin/env python3
"""Four configurations, three frozen datasets, one resident model per A100 worker."""
from __future__ import annotations

from argparse import ArgumentParser
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
import gc
from hashlib import sha256
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

from specfit.experiment import canonical_json_sha256
from specfit.checkpoint_comparison import CHAIN_METHODS, CONFIG_SHA256, QUESTION_IDS, NativeChainRecorder
from specfit.main_comparison import (PROTOCOL, METHODS, REFERENCE_METHODS, NO_CACHE, TOTAL_TOKENS, LEVEL_METHODS,
    LevelSpecFit, SMOKE_IDS, PILOT_IDS, GPU_UUIDS, GPU_NAME, DEVICES, choose, load_prompt_budget, load_workloads, numeric_grade,
    reference_method, verify_record, summarize, predicted_routes, ROUTE_SEED, RevisedCostModel, RevisedLevelSpecFit)
from specfit.r6a_tb2_contract import R6A_TB2_IMAGE_ID
from specfit.selection_methods import SelectionCostModel
from specfit.serial_replay import COUNTERS, file_identity
from run_qwen3_serial_replay import cache_boundary, read_json, write_json
from specfit_main_recovery import prepare_recovery, prepare_reference_reuse, reused_stream

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_DEPTH = ('after drafting level d < 5 (or the tail limit), stop when SpecFit '
                  'on the depth<=d pool selects no depth-d node; its selection is final')
# Recorded A100 calibration: the cost model of every A100 run, and the transfer ablation on the second device.
A100_CALIBRATION = ROOT / 'docs/results/selection-refinement-1886852.json'
DEVICE_METHODS = ('ar', 'ar_no_cache', 'eagle_no_cache', 'eagle_cache', 'specfit', 'specfit_no_cache',
                  'specfit_adaptive', 'specfit_adaptive_no_cache')


def arguments():
    p = ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'run', 'validate'))
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--prepared-root', type=Path)
    p.add_argument('--git-commit')
    p.add_argument('--image-id', default=R6A_TB2_IMAGE_ID)
    p.add_argument('--worker', type=int, choices=(0, 1), default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--checkpoint-default-comparison', action='store_true')
    p.add_argument('--slot-capacity', type=int)
    p.add_argument('--reference-from', type=Path)
    p.add_argument('--summarization', action='store_true')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--adaptive', action='store_true',
                   help='SpecFit with adaptive draft depth: pilot, 40 summarization articles, or GSM8K (with --reference-from)')
    p.add_argument('--complete-table', action='store_true',
                   help='with --adaptive --summarization --reference-from: AR with cache, EAGLE and adaptive SpecFit without cache')
    p.add_argument('--device', choices=tuple(DEVICES),
                   help='second device: fresh GSM8K run of --methods with its own references and --device-slots')
    p.add_argument('--methods', nargs='+', choices=DEVICE_METHODS)
    p.add_argument('--device-slots', type=int)
    p.add_argument('--cost-model', type=Path,
                   help='cost model of the device\'s SpecFit methods: its refit, or the recorded A100 calibration (transfer ablation)')
    p.add_argument('--cold-cost-model', action='store_true',
                   help='with --adaptive: adaptive SpecFit without cache on GSM8K (--slot-capacity 0) or on the 40 '
                        'summarization articles selects with the cold-cache coefficients; fresh reference. Second device, '
                        'SpecFit without cache only: select with the cold-cache coefficients of the cost model')
    p.add_argument('--route-accuracy', type=float,
                   help='with --adaptive: GSM8K (even question IDs) adaptive SpecFit selecting with simulated predictor '
                        'routes of this per-expert accuracy; fresh reference')
    p.add_argument('--revised-cost', type=Path,
                   help='with --route-accuracy 1.0, or a second-device stage of cached adaptive SpecFit next to its refit '
                        'cost model: select with the revised cost model fitted in this file')
    return p.parse_args()


def freeze(args):
    from benchmark_qwen3_selection import head
    if head() != args.git_commit or args.image_id != R6A_TB2_IMAGE_ID:
        raise ValueError('clean frozen source and fixed image required')
    adaptive_gsm8k = (args.adaptive and not args.pilot and not args.summarization and args.slot_capacity is None
                      and args.route_accuracy is None)
    # Complements of the adaptive paper runs: measured streams only, references reused from the sealed run on this GPU.
    complement = args.adaptive and not args.cold_cost_model and (args.slot_capacity is not None or args.complete_table)
    if args.cold_cost_model and args.device is None and not (args.adaptive and not args.pilot and not args.complete_table and args.reference_from is None
                                     and args.slot_capacity == (None if args.summarization else 0)):
        raise ValueError('the cold-cache cost model is the fresh zero-slot GSM8K run or the fresh no-cache summarization '
                         'row of adaptive SpecFit')
    if args.route_accuracy is not None and not (args.adaptive and not args.pilot and not args.summarization
            and not args.complete_table and not args.cold_cost_model and args.slot_capacity is None
            and args.reference_from is None and 0 < args.route_accuracy <= 1):
        raise ValueError('route-accuracy runs are fresh 48-slot GSM8K runs of adaptive SpecFit')
    if args.revised_cost is not None and args.device is None and args.route_accuracy != 1.0:
        raise ValueError('the revised cost model runs as a variant of the matched exact-route control')
    if args.complete_table and not (args.adaptive and args.summarization and args.slot_capacity is None):
        raise ValueError('the complete table complements the adaptive summarization run')
    if args.device is not None and (args.adaptive or args.summarization or args.pilot or args.smoke or args.complete_table
            or args.checkpoint_default_comparison or args.slot_capacity is not None or args.reference_from is not None
            or args.resume_from is not None or args.worker != 0):
        raise ValueError('the second-device run is a fresh GSM8K profile with its own references')
    if (args.device is None) != (args.methods is None and args.device_slots is None and args.cost_model is None):
        raise ValueError('--methods, --device-slots and --cost-model belong to the second-device run')
    names = ('summarization',) if args.summarization else () if args.adaptive or args.device else ('humaneval', 'aime25')
    work, identities = load_workloads(args.dataset, args.prepared_root, ROOT, names)
    calibration = args.cost_model or A100_CALIBRATION
    mode = 'cold' if args.cold_cost_model else 'natural'
    model = read_json(calibration)['stages']['calibration']['models']['by_mode'][mode]['legacy']
    SelectionCostModel.from_dict(model)
    order = list(METHODS) if args.worker == 0 else list(reversed(METHODS))
    frozen = dict(protocol=PROTOCOL, git_commit=args.git_commit, image_id=args.image_id,
        original_method_commit='c02847726263ee3ca886544d2e93c6561124ec96',
        worker=args.worker, gpu_uuid=GPU_UUIDS[args.worker], smoke=args.smoke,
        datasets={name: dict(total_tokens=TOTAL_TOKENS[name],
            question_ids=list(SMOKE_IDS[name]) if args.smoke else [r['question_id'] for r in rows],
            identity=identities[name]) for name, rows in work.items()},
        dataset_order=['gsm8k', 'humaneval', 'aime25'], method_order=order,
        methods=list(METHODS), reference_methods=list(REFERENCE_METHODS),
        repetitions_per_worker=1, repetitions_across_workers=2,
        cost_model=model, calibration=file_identity(calibration),
        execution='ready_demand', slots_per_layer=48, max_inflight=2,
        tree=dict(top_k=2, max_depth=5, pool_node_budget=18, eagle_draft_node_budget=7,
                  specfit_draft_node_budget=[0, 7]),
        cache_policy='cold_start_per_dataset_method_stream_then_preserve; eagle_no_cache_discards_before_every_target_forward',
        route_information='offline_full_pool_atlas_from_separate_audited_generation',
        max_wall_seconds=4*3600 if args.smoke else 72*3600, max_request_seconds=3600,
        max_gpu_reserved_bytes=38*1024**3, max_rss_bytes=128*1024**3,
        timing_scope='request includes prefill/draft/selection/verify/commit/cache-discard; excludes loading/reference/atlas/hash/file-write/quality-scoring',
        limitations=['EAGLE-3 via speculators and project offload backend, not official default tree parameters.',
            'SpecFit offline route acquisition excluded; not online end-to-end performance.',
            'Same-query repeat verified; cross-tree and AR output equality not assumed.',
            'One complete repeat on each physical A100; compare all methods within GPU and aggregate token/time.'])
    if args.checkpoint_default_comparison:
        if args.smoke or args.resume_from is not None:
            raise ValueError('checkpoint comparison is an independent small experiment')
        methods = ['ar', *CHAIN_METHODS, 'eagle_no_cache', 'eagle_cache', 'specfit']
        frozen.update(profile='checkpoint-default-gsm8k-v1',
            methods=methods, method_order=methods if args.worker == 0 else list(reversed(methods)),
            reference_methods=['ar', 'eagle_chain3_cache', 'eagle_cache', 'specfit'],
            datasets={'gsm8k': dict(frozen['datasets']['gsm8k'], question_ids=list(QUESTION_IDS))},
            dataset_order=['gsm8k'], max_wall_seconds=2*3600,
            checkpoint_proposal=dict(config_sha256=CONFIG_SHA256, proposal_type='greedy', speculative_tokens=3,
                implementation='Eagle3ChainDrafter -> checkpoint eager forward(ttt_steps=3); full prefix each cycle'))
        frozen['limitations'] += ['Four predetermined GSM8K questions are diagnostic, not a full benchmark.',
            'Native chain proposal recomputes its prefix; tree proposal uses an incremental draft KV cache.',
            'Chain audit checks target argmax acceptance and actual KV length; it does not inspect auxiliary-history tensors.']
    if args.pilot and (not args.summarization or args.smoke or args.resume_from is not None):
        raise ValueError('pilot is a fresh summarization run')
    if args.summarization:
        if args.checkpoint_default_comparison or (not complement and (args.slot_capacity is not None or args.reference_from is not None)):
            raise ValueError('summarization comparison is an independent profile with its own references')
        frozen.update(profile='summarization-specbench-v1', datasets={'summarization': frozen['datasets']['summarization']},
            dataset_order=['summarization'], method_order=list(METHODS), repetitions_across_workers=1,
            max_wall_seconds=2*3600 if args.smoke else 14*3600)
        frozen['limitations'] = [x for x in frozen['limitations'] if not x.startswith('One complete repeat on each physical A100')]
        frozen['limitations'] += ['Spec-Bench CNN/DailyMail summarization (80 articles) with its verbatim prompt; summaries are not scored, only efficiency is compared.',
            'Cost coefficients calibrated on GSM8K decoding with 48 slots are not refit for long summarization contexts.',
            f'One complete repeat on GPU {args.worker} only (method order AR, EAGLE without/with cache, SpecFit).',
            'The other A100 may run an independent offloaded job; the host controller records the packed-cache copy and both GPUs.']
    if args.pilot:
        methods = ['eagle_cache', 'specfit']
        frozen.update(profile='summarization-pilot-v1', methods=methods, method_order=methods, reference_methods=methods,
            datasets={'summarization': dict(frozen['datasets']['summarization'], question_ids=list(PILOT_IDS['summarization']))},
            max_wall_seconds=2*3600)
        frozen['limitations'] = [x.replace('(method order AR, EAGLE without/with cache, SpecFit)', '(method order EAGLE with cache, SpecFit)')
                                 for x in frozen['limitations']]
        frozen['limitations'] += ['Pilot: eight predetermined articles spread over input length, EAGLE with cache and SpecFit only; diagnostic, not the benchmark.']
    if args.adaptive:
        if args.smoke or args.resume_from is not None or args.checkpoint_default_comparison:
            raise ValueError('adaptive draft depth runs are fresh profiles')
        frozen['tree'].update(specfit_adaptive_depth=ADAPTIVE_DEPTH)
    if args.adaptive and args.pilot:
        methods = ['eagle_cache', 'specfit', *LEVEL_METHODS]
        frozen.update(profile='summarization-adaptive-pilot-v1', methods=methods, method_order=methods,
            reference_methods=['eagle_cache', 'specfit', 'specfit_adaptive'])
        frozen['limitations'] = [x.replace('(method order EAGLE with cache, SpecFit)',
                                           '(method order EAGLE with cache, SpecFit, packed SpecFit, adaptive SpecFit)')
                                  .replace('EAGLE with cache and SpecFit only', 'EAGLE with cache and three SpecFit variants only')
                                 for x in frozen['limitations']]
        frozen['limitations'] += ['Packed SpecFit makes the same choices as SpecFit with the packed exact search and shares its '
            'reference; it isolates selector cost. Adaptive SpecFit uses that search at every drafted level, with routes of the '
            'depth<=d prefix taken from its separate reference that probes the full 18-node pool.']
    if args.adaptive and args.summarization and not args.pilot:
        # Paper run: every other article; AR without expert cache is the speedup baseline.
        methods = ['ar_no_cache', 'eagle_cache', 'specfit', 'specfit_adaptive']
        spec = frozen['datasets']['summarization']
        frozen.update(profile='summarization-adaptive-v1', methods=methods, method_order=methods,
            reference_methods=[reference_method(m) for m in methods],
            datasets={'summarization': dict(spec, question_ids=spec['question_ids'][::2])},
            cache_policy='cold_start_per_dataset_method_stream_then_preserve; ar_no_cache discards completed experts before every target forward',
            max_wall_seconds=8*3600)
        frozen['limitations'] = [x.replace('(80 articles)', '(40 of 80 articles: even question IDs)')
                                  .replace('(method order AR, EAGLE without/with cache, SpecFit)',
                                           '(method order AR without cache, EAGLE with cache, SpecFit, adaptive SpecFit)')
                                 for x in frozen['limitations']]
    if adaptive_gsm8k:
        if args.reference_from is None:
            raise ValueError('the GSM8K adaptive run reuses the sealed 48-slot SpecFit references')
        # Paper run: adaptive SpecFit beside a re-measured fixed-depth SpecFit on the same GPU.
        methods = ['specfit_adaptive', 'specfit']
        frozen.update(profile='adaptive-gsm8k-v1', methods=methods, method_order=methods,
            reference_methods=['specfit', 'specfit_adaptive'], reused_reference_methods=['specfit'],
            datasets={'gsm8k': frozen['datasets']['gsm8k']}, dataset_order=['gsm8k'],
            repetitions_across_workers=1, max_wall_seconds=4*3600)
        frozen['limitations'] = [x for x in frozen['limitations'] if not x.startswith('One complete repeat on each physical A100')]
        frozen['limitations'] += [f'One complete repeat on GPU {args.worker} only (method order adaptive SpecFit, SpecFit).',
            'SpecFit references come from the sealed 48-slot run; each measured output and per-forward logits/route digest must match them.']
    if args.adaptive and not args.pilot:
        frozen['limitations'] += ['Adaptive SpecFit uses the packed exact search at every drafted level, with routes of the '
            'depth<=d prefix taken from its separate reference that probes the full 18-node pool.']
    if complement:
        slots = args.slot_capacity
        if args.pilot or args.reference_from is None:
            raise ValueError('adaptive complements reuse the sealed references of an adaptive paper run')
        if args.summarization:
            if slots is not None and not 8 <= slots < 48:
                raise ValueError('the summarization slot sweep uses 8-47 retained slots; the complete table holds the no-cache rows')
            # Beside the 40-article run: its missing table rows, or EAGLE vs adaptive SpecFit with a reduced cache.
            methods = ['ar', 'eagle_no_cache', 'specfit_adaptive_no_cache'] if slots is None else ['eagle_cache', 'specfit_adaptive']
            order = ('(method order AR with cache, EAGLE without cache, adaptive SpecFit without cache)' if slots is None
                     else '(method order EAGLE with cache, adaptive SpecFit)')
            frozen.update(profile='summarization-adaptive-complete-v1' if slots is None else 'summarization-adaptive-slots-v1',
                          max_wall_seconds=6*3600)
            frozen['limitations'] = [x.replace('(method order AR without cache, EAGLE with cache, SpecFit, adaptive SpecFit)', order)
                                     for x in frozen['limitations']]
        else:
            if not (slots == 0 or 8 <= slots < 48):
                raise ValueError('the GSM8K adaptive slot sweep uses 0 or 8-47 retained slots')
            # EAGLE and fixed-depth SpecFit at the same slots are the sealed slot-sweep streams on this GPU.
            methods = ['specfit_adaptive_no_cache'] if slots == 0 else ['specfit_adaptive']
            frozen.update(profile='adaptive-slot-capacity-gsm8k-v1', datasets={'gsm8k': frozen['datasets']['gsm8k']},
                dataset_order=['gsm8k'], repetitions_across_workers=1, max_wall_seconds=3*3600)
            frozen['limitations'] = [x for x in frozen['limitations'] if not x.startswith('One complete repeat on each physical A100')]
            frozen['limitations'] += [f'One adaptive SpecFit stream on GPU {args.worker}; EAGLE and SpecFit at the same slots '
                                      'come from the sealed slot sweep on this GPU.']
        frozen.update(methods=methods, method_order=methods, reference_methods=[reference_method(m) for m in methods])
        if slots is not None:
            frozen.update(slots_per_layer=slots or 48, retained_expert_slots=slots)
            frozen['limitations'] += ['Fewer expert slots per layer on an A100 emulate tighter device memory; not a consumer-device measurement.',
                'Cost coefficients calibrated with 48 slots are not refit for the reduced capacity.']
        if slots == 0:
            frozen['cache_policy'] = 'cold_start_per_dataset_method_stream; every method discards completed experts before every target forward'
        elif slots is None:
            frozen['cache_policy'] = ('cold_start_per_dataset_method_stream_then_preserve; '
                                      'no-cache methods discard completed experts before every target forward')
        else:
            frozen['cache_policy'] = 'cold_start_per_dataset_method_stream_then_preserve'
        frozen['limitations'] += ['References come from the sealed 48-slot adaptive run on this GPU; each measured output '
                                  'and per-forward logits/route digest must match them.']
    if args.cold_cost_model and args.summarization:
        # No-cache row of the 40-article table: experts are loaded before every target forward, so adaptive SpecFit
        # selects with the cold-cache coefficients; its trees differ from the natural-cache run and get their own reference.
        methods = ['specfit_adaptive_no_cache']
        frozen.update(profile='summarization-adaptive-cold-v1', cost_model_mode=mode, methods=methods, method_order=methods,
            reference_methods=['specfit_adaptive'], max_wall_seconds=4*3600,
            cache_policy='cold_start_per_dataset_method_stream; every method discards completed experts before every target forward')
        frozen['limitations'] = [x.replace('(method order AR without cache, EAGLE with cache, SpecFit, adaptive SpecFit)',
                                           '(one stream: adaptive SpecFit without cache)') for x in frozen['limitations']]
        frozen['limitations'] += ['Adaptive SpecFit selects with the cold-cache legacy coefficients of the calibration; AR, EAGLE '
                                  'and fixed-depth SpecFit come from the sealed 40-article runs on this GPU.']
    elif args.cold_cost_model:
        # Zero-slot paper row: experts are loaded before every target forward, so adaptive SpecFit selects with the
        # cold-cache coefficients; its trees differ from the natural-cache runs and get their own reference.
        methods = ['specfit_adaptive_no_cache']
        frozen.update(profile='adaptive-cold0-gsm8k-v2', cost_model_mode=mode, methods=methods, method_order=methods,
            reference_methods=['specfit_adaptive'], datasets={'gsm8k': frozen['datasets']['gsm8k']},
            dataset_order=['gsm8k'], repetitions_across_workers=1, slots_per_layer=48, retained_expert_slots=0,
            max_wall_seconds=4*3600,
            cache_policy='cold_start_per_dataset_method_stream; every method discards completed experts before every target forward')
        frozen['limitations'] = [x for x in frozen['limitations'] if not x.startswith('One complete repeat on each physical A100')]
        frozen['limitations'] += [f'One complete repeat on GPU {args.worker} only (one stream: adaptive SpecFit without cache).',
            'Adaptive SpecFit selects with the cold-cache legacy coefficients of the calibration; AR and EAGLE without cache '
            'come from the sealed zero-slot slot sweep on this GPU.',
            'Zero retained slots discards experts before every target forward, as the main-table EAGLE no-cache row.']
    if args.route_accuracy is not None:
        # Simulated expert predictor: selection and the depth decision see routes with a fraction of each token's
        # experts replaced; verification runs the true routing, so outputs stay lossless and only the trees change.
        methods = ['specfit_adaptive']
        spec = frozen['datasets']['gsm8k']
        frozen.update(profile='route-accuracy-gsm8k-v1', route_accuracy=args.route_accuracy, route_seed=ROUTE_SEED,
            route_information='offline_full_pool_atlas_from_separate_audited_generation; selection sees simulated predictor routes',
            methods=methods, method_order=methods, reference_methods=methods,
            datasets={'gsm8k': dict(spec, question_ids=spec['question_ids'][1::2])}, dataset_order=['gsm8k'],
            repetitions_across_workers=1, max_wall_seconds=3*3600)
        frozen['limitations'] = [x for x in frozen['limitations'] if not x.startswith('One complete repeat on each physical A100')]
        frozen['limitations'] += [f'One adaptive SpecFit stream on GPU {args.worker} over 40 of the 80 GSM8K questions '
            '(even question IDs); exact routes, EAGLE and AR come from the sealed full-length streams on this GPU.',
            f'Simulated predictor: each routed expert of every node and layer is kept with probability {args.route_accuracy}, '
            'otherwise replaced by a uniformly drawn expert outside its true top-8; real predictor errors are not uniform.']
    if args.revised_cost is not None and args.device is None:
        # Exact routes, selected with the cost model refit on the odd-question cycles of the sealed main-table stream.
        fitted = read_json(args.revised_cost)
        RevisedCostModel.from_dict(fitted['model'])
        frozen.update(profile='revised-cost-gsm8k-v1', revised_cost_model=fitted['model'],
                      revised_cost=file_identity(args.revised_cost))
        frozen['limitations'] = [x for x in frozen['limitations'] if not x.startswith('Simulated predictor:')]
        frozen['limitations'] += ['Revised cost model: intercepts and slope refit on the odd-question cycles of the '
            'sealed main-table stream on GPU 0, proposal time by drafted levels; a tree using a node of the level just '
            'drafted is charged the next level.']
    if args.slot_capacity is not None and not args.adaptive:
        if (args.smoke or args.resume_from is not None or args.checkpoint_default_comparison
                or args.reference_from is None or not (args.slot_capacity == 0 or 8 <= args.slot_capacity <= 48)):
            raise ValueError('slot sweep uses 0 or 8-48 retained slots and reuses sealed 48-slot references only')
        # Zero retained slots is the main-table no-cache setting: 48 slots hold only one forward's working set.
        methods = ['ar_no_cache', 'eagle_no_cache', 'specfit_no_cache'] if args.slot_capacity == 0 else ['eagle_cache', 'specfit']
        frozen.update(profile='slot-capacity-gsm8k-v1', slots_per_layer=args.slot_capacity or 48,
            retained_expert_slots=args.slot_capacity,
            methods=methods, method_order=methods if args.worker == 0 else list(reversed(methods)),
            reference_methods=[reference_method(m) for m in methods], datasets={'gsm8k': frozen['datasets']['gsm8k']},
            dataset_order=['gsm8k'], max_wall_seconds=4*3600)
        frozen['limitations'] += ['Fewer expert slots per layer on an A100 emulate tighter device memory; not a consumer-device measurement.',
            'Cost coefficients calibrated with 48 slots are not refit for the reduced capacity.',
            'References come from the sealed 48-slot run; each measured output and per-forward logits/route digest must match them.']
        if args.slot_capacity == 0:
            frozen['cache_policy'] = 'cold_start_per_dataset_method_stream; every method discards completed experts before every target forward'
            frozen['limitations'] += ['Zero retained slots discards experts before every target forward, as the main-table EAGLE no-cache row; AR without cache is the speedup baseline.']
    elif args.reference_from is not None and not (adaptive_gsm8k or complement):
        raise ValueError('reference reuse is limited to the slot sweep and the adaptive runs')
    if args.device is not None:
        device_profile(args, frozen)
    args.directory.mkdir(parents=True, exist_ok=False)
    if args.reference_from is not None:
        frozen['recovery'] = prepare_reference_reuse(args.reference_from, args.directory, frozen)
    if args.resume_from is not None:
        frozen['recovery'] = prepare_recovery(args.resume_from, args.directory, frozen, ROOT)
    write_json(args.directory/'manifest.json', frozen)
    print(json.dumps(frozen, ensure_ascii=False), flush=True)


def device_profile(args, frozen):
    """Second device: its own references, slot count and (for SpecFit) cost model; every other GSM8K question."""
    methods, slots = args.methods or [], args.device_slots
    references = list(dict.fromkeys(reference_method(m) for m in methods))
    selects = any(m in ('specfit', 'specfit_adaptive') for m in references)
    if (not methods or len(set(methods)) != len(methods) or slots is None or not 8 <= slots < 48
            or selects != (args.cost_model is not None)):
        raise ValueError('the second-device run needs distinct methods, 8-47 slots, and a cost model exactly when SpecFit selects')
    source = None
    if selects:
        # Either the refit written by the device calibration, or the A100 coefficients unchanged (transfer ablation).
        record = read_json(args.cost_model)
        if record.get('artifact_type') == 'device_cost_model' and record.get('device') == args.device:
            source = 'refit_on_device'
        elif file_identity(args.cost_model)['sha256'] == file_identity(A100_CALIBRATION)['sha256']:
            source = 'a100_calibration'
        else:
            raise ValueError('the cost model is neither this device\'s refit nor the recorded A100 calibration')
    # Without the expert cache every forward loads its experts, so exactly those configs select with the cold-cache fit.
    no_cache = {'specfit_no_cache', 'specfit_adaptive_no_cache'}
    if (args.cold_cost_model and not (methods and set(methods) <= no_cache)
            or not args.cold_cost_model and set(methods) & no_cache):
        raise ValueError('SpecFit without cache selects with the cold-cache cost model, in a stage of its own')
    spec = frozen['datasets']['gsm8k']
    frozen.update(profile=args.device+'-gsm8k-v1', device=args.device, **DEVICES[args.device],
        methods=methods, method_order=methods, reference_methods=references,
        datasets={'gsm8k': dict(spec, question_ids=spec['question_ids'][::2])}, dataset_order=['gsm8k'],
        slots_per_layer=slots, retained_expert_slots=slots, repetitions_across_workers=1,
        max_wall_seconds=12*3600, max_gpu_reserved_bytes=23*1024**3,
        cache_policy='cold_start_per_dataset_method_stream_then_preserve; '
                     'no-cache methods discard completed experts before every target forward')
    if 'specfit_adaptive' in references:
        frozen['tree'].update(specfit_adaptive_depth=ADAPTIVE_DEPTH)
    if source is not None:
        frozen.update(cost_model_source=source, cost_model_mode='cold' if args.cold_cost_model else 'natural')
    frozen['limitations'] = [x for x in frozen['limitations'] if not x.startswith('One complete repeat on each physical A100')]
    frozen['limitations'] += [
        f'One repeat on one {frozen["gpu_name"]} (24GB, PCIe Gen3 x16) on another host; method order {", ".join(methods)}.',
        f'GSM8K every other question ({len(frozen["datasets"]["gsm8k"]["question_ids"])} of 80); {slots} expert slots per layer, '
        'the 48 slots of the A100 runs do not fit this memory.',
        'References are fresh on this GPU; each measured output and per-forward logits/route digest must match them.',
        {'refit_on_device': 'Cost coefficients refit on this GPU with the frozen calibration protocol.',
         'a100_calibration': 'Cost-model transfer: SpecFit selects with the recorded A100 coefficients, not refit on this GPU.',
         None: 'No SpecFit selection: the recorded A100 cost model is not used.'}[source]]
    if args.cold_cost_model:
        frozen['limitations'].append('SpecFit without cache selects with the cold-cache legacy coefficients of that calibration, '
                                     'so its specfit_adaptive/specfit reference is generated with them.')
    if args.revised_cost is not None:
        # Cached adaptive SpecFit selecting with the revised cost model refit on this device's sealed natural-fit stream.
        fitted = read_json(args.revised_cost)
        RevisedCostModel.from_dict(fitted['model'])
        if methods != ['specfit_adaptive'] or source != 'refit_on_device' or fitted.get('device') != args.device:
            raise ValueError('the revised cost model is a stage of cached adaptive SpecFit fitted on this device')
        frozen.update(revised_cost_model=fitted['model'], revised_cost=file_identity(args.revised_cost),
                      revised_cost_source=dict(stage=fitted['stage'], manifest_sha256=fitted['manifest_sha256'],
                                               question_ids=fitted['question_ids'], cycles=fitted['cycles'],
                                               skipped_leading_cycles=fitted['skipped_leading_cycles'],
                                               intercept=fitted['intercept']))
        frozen['limitations'].append('Revised cost model: refit on the cycles of the sealed natural-fit adaptive SpecFit stream '
            'of this device on the same questions (in-sample), intercept linear in the node count, proposal time by drafted '
            'levels; a tree using a node of the level just drafted is charged the next level.')


def validate(directory, *, persist=False):
    frozen, index = read_json(directory/'manifest.json'), read_json(directory/'result.json')
    if (index['status'] != 'complete' or index['protocol'] != PROTOCOL
            or index['manifest_sha256'] != canonical_json_sha256(frozen)
            or index['git_commit'] != frozen['git_commit']):
        raise ValueError('incomplete or mismatched experiment')
    model = SelectionCostModel.from_dict(frozen['cost_model'])
    revised = frozen.get('revised_cost_model')
    revised = None if revised is None else RevisedCostModel.from_dict(revised)
    refs, summaries, measured = {}, [], {}
    for descriptor in index['references']:
        row = read_json(directory/descriptor['basename'], descriptor)
        key = row['dataset'], row['method'], row['question_id']
        if key in refs or row['method'] not in frozen['reference_methods']:
            raise ValueError('duplicate/unknown reference')
        verify_record(row, None, model, frozen['datasets'][row['dataset']]['total_tokens'], frozen.get('route_accuracy'),
                      revised)
        refs[key] = row
    expected = {(dataset, method, q) for dataset, spec in frozen['datasets'].items()
                for method in frozen['reference_methods'] for q in spec['question_ids']}
    if set(refs) != expected:
        raise ValueError('incomplete reference coverage')
    streams = {}
    for descriptor in index['questions']:
        row = read_json(directory/descriptor['basename'], descriptor)
        key = row['dataset'], row['method'], row['question_id']
        if key in measured or row['repetition'] != frozen['worker']:
            raise ValueError('duplicate measurement or wrong repeat')
        reference = refs[row['dataset'], reference_method(row['method']), row['question_id']]
        verify_record(row, reference, model, frozen['datasets'][row['dataset']]['total_tokens'], frozen.get('route_accuracy'),
                      revised)
        measured[key] = row
        streams.setdefault((row['dataset'], row['method']), []).append(row)
    expected = {(dataset, method, q) for dataset, spec in frozen['datasets'].items()
                for method in frozen['methods'] for q in spec['question_ids']}
    if set(measured) != expected:
        raise ValueError('incomplete measurement coverage')
    for dataset in frozen['dataset_order']:
        for method in frozen['methods']:
            rows = streams[dataset, method]
            if [r['question_id'] for r in rows] != frozen['datasets'][dataset]['question_ids'] or rows[0]['pre_cache_occupied'] != 0:
                raise ValueError('stream order/cold start changed')
            if any(b['pre_cache_sha256'] != a['post_cache_sha256'] for a, b in zip(rows, rows[1:])):
                raise ValueError('cache continuity changed')
            for row in rows:
                # Without the cached counterpart in this run, verify_record already matched the reference.
                if method in NO_CACHE and (dataset, NO_CACHE[method], row['question_id']) in measured:
                    other = measured[dataset, NO_CACHE[method], row['question_id']]
                    if any(row[k] != other[k] for k in ('output_token_ids', 'query_proofs', 'cycles')):
                        raise ValueError('cache switch changed mathematical queries/output')
            summaries.append(dict(dataset=dataset, method=method, worker=frozen['worker'], **summarize(rows)))
    report = dict(status='verified', protocol=PROTOCOL, references=len(refs), measured_requests=len(measured),
                  manifest_sha256=index['manifest_sha256'], summaries=summaries)
    if persist:
        write_json(directory/'validation.json', report)
    return report


def run(args):
    import torch
    from transformers import AutoTokenizer
    import benchmark_qwen3_m1_full_atlas as m1
    from benchmark_qwen3_selection import head
    from specfit.r6a_runner import serialize_candidate_pool
    from specfit.main_comparison_runtime import discard_completed_experts
    from specfit.runtime import (Eagle3ChainDrafter, Eagle3TreeDrafter, ExplicitRawIdTreeSelector,
        ExpertExecutionPolicy, ReductionMode, ShallowTreeSpeculator, apply_numeric_mode,
        load_offloaded_qwen3, target_greedy_generate, warmup_eagle3_tree_drafter, candidate_pool_fingerprint,
        GreedyChainSpeculator)
    from specfit.runtime.eagle3_chain import _experimental_2048_token_budget
    from specfit.runtime.eagle3_tree import candidate_pool_prefix

    frozen = read_json(args.directory/'manifest.json')
    if head() != frozen['git_commit'] or args.git_commit != frozen['git_commit'] or args.image_id != frozen['image_id']:
        raise ValueError('source/image changed')
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != frozen.get('gpu_name', GPU_NAME)
            or str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') != frozen['gpu_uuid'].removeprefix('GPU-')):
        raise ValueError('physical GPU changed')
    work, identities = load_workloads(args.dataset, args.prepared_root, ROOT,
                                      tuple(n for n in frozen['datasets'] if n != 'gsm8k'))
    if any(identities[n] != d['identity'] for n, d in frozen['datasets'].items()):
        raise ValueError('input bytes changed')
    source = m1._source_preflight(args)
    if 'checkpoint_proposal' in frozen:
        config_path = Path(m1._DRAFTER)/'config.json'
        config = read_json(config_path)['speculators_config']
        if (sha256(config_path.read_bytes()).hexdigest() != frozen['checkpoint_proposal']['config_sha256']
                or config['default_proposal_method'] != 'greedy'
                or config['proposal_methods'] != [dict(accept_tolerance=0.0, proposal_type='greedy',
                                                       speculative_tokens=3, verifier_accept_k=1)]):
            raise ValueError('checkpoint proposal defaults changed')
    tokenizer = AutoTokenizer.from_pretrained(m1._TARGET, local_files_only=True)
    prepared = {}
    prompt_lock = load_prompt_budget(ROOT)
    for name, digest in prompt_lock['tokenizer_files'].items():
        if sha256((Path(m1._TARGET)/name).read_bytes()).hexdigest() != digest:
            raise ValueError('tokenizer changed')
    for dataset in frozen['dataset_order']:
        rows = work[dataset]
        selected = set(frozen['datasets'][dataset]['question_ids'])
        expected = {r['question_id']: r for r in prompt_lock['datasets'].get(dataset, {}).get('requests', [])}
        for row in rows:
            if row['question_id'] not in selected:
                continue
            prompt = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids
            digest = canonical_json_sha256(ids.flatten().tolist())
            if not 0 < ids.shape[1] < TOTAL_TOKENS[dataset] or (expected and digest != expected[row['question_id']]['input_sha256']):
                raise ValueError('prompt differs from audited input or exceeds budget')
            prepared[dataset, row['question_id']] = dict(row, rendered_prompt=prompt, ids=ids, input_ids_sha256=digest)
    model = SelectionCostModel.from_dict(frozen['cost_model'])
    accuracy = frozen.get('route_accuracy')
    revised = frozen.get('revised_cost_model')
    revised = None if revised is None else RevisedCostModel.from_dict(revised)

    def level_selector(atlas):
        return LevelSpecFit(model, atlas) if revised is None else RevisedLevelSpecFit(revised, atlas)
    index = dict(protocol=PROTOCOL, status='running', git_commit=frozen['git_commit'],
        manifest_sha256=canonical_json_sha256(frozen), started_ns=time.time_ns(),
        environment=dict(image_id=args.image_id, models=source, gpu_uuid=frozen['gpu_uuid']), references=[], questions=[])
    write_json(args.directory/'started.json', index)
    loaded = numeric = None
    began = time.perf_counter()
    request_start = None
    gc_enabled = gc.isenabled()

    def event(**fields):
        print(json.dumps(dict(time_ns=time.time_ns(), worker=frozen['worker'], **fields)), flush=True)

    def budget():
        now = time.perf_counter()
        if (now-began > frozen['max_wall_seconds'] or (request_start is not None and now-request_start > frozen['max_request_seconds'])
                or torch.cuda.max_memory_reserved() > frozen['max_gpu_reserved_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > frozen['max_rss_bytes']):
            raise RuntimeError('frozen time/resource limit exceeded')

    def digest(tensor):
        return sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    class Execution:
        def __init__(self, method):
            self.method, self.decode, self.probing = method, False, False
            self.proofs, self.routes = [], {}
            self.discard_calls = self.discard_ns = 0

        def hook(self, lid):
            def capture(module, inputs, output):
                self.routes[lid] = output[2].detach()
            return capture

        @contextmanager
        def observe(self):
            original = loaded.model.forward
            def forward(*pos, **kw):
                budget()
                self.routes = {}
                if self.method in NO_CACHE:
                    start = time.perf_counter_ns()
                    discard_completed_experts(loaded)
                    self.discard_ns += time.perf_counter_ns()-start
                    self.discard_calls += 1
                if self.decode:
                    with loaded.prefetch_experts({}, lookahead_layers=0, max_inflight=2):
                        output = original(*pos, **kw)
                else:
                    output = original(*pos, **kw)
                if not self.probing:
                    self.proofs.append((output.logits.detach(), tuple(self.routes[i] for i in range(48))))
                if self.method in CHAIN_METHODS:
                    self.last_cache = output.past_key_values
                return output
            with ExitStack() as stack:
                loaded.model.forward = forward
                stack.callback(setattr, loaded.model, 'forward', original)
                for lid, layer in enumerate(loaded.model.model.layers):
                    handle = layer.mlp.gate.register_forward_hook(self.hook(lid))
                    stack.callback(handle.remove)
                yield

        def materialize(self):
            result = []
            for logits, routes in self.proofs:
                route_rows = [r.cpu().tolist() for r in routes]
                union = sum(torch.unique(r).numel() for r in routes)
                result.append(dict(logits_sha256=digest(logits), logits_shape=list(logits.shape),
                    routes_sha256=canonical_json_sha256(route_rows), expert_union=union,
                    finite=bool(torch.isfinite(logits).all().item())))
                if self.method in CHAIN_METHODS:
                    result[-1]['greedy_tokens'] = logits.argmax(dim=-1).flatten().cpu().tolist()
            return result

    class Selector:
        def __init__(self, method, execution, reference, key):
            self.method, self.execution, self.reference, self.key = method, execution, reference, key
            self.boundary, self.rows = None, []
            self.level = self.stopped = None
            # Simulated predictor routes of the measured stream, prepared before the request clock starts.
            self.predicted = (None if accuracy is None or reference is None else
                [predicted_routes(c['atlas'], accuracy, (*key, i)) for i, c in enumerate(reference['cycles'])])

        def at_boundary(self, checkpoint):
            self.boundary = checkpoint
            budget()

        def probe(self, pool):
            """Reference streams only: route every candidate of the full pool through the target."""
            state = self.boundary.state
            tree = pool.to_tree(tuple(n.raw_id for n in pool.candidates))
            inputs = tree.verification_token_ids(state.pending_seed)
            cache = deepcopy(state.cache)
            self.execution.probing = True
            try:
                output = loaded.model(input_ids=inputs,
                    attention_mask=tree.verification_attention_mask(base_kv_length=cache.get_seq_length(), device=inputs.device),
                    position_ids=tree.verification_position_ids(base_kv_length=cache.get_seq_length(), device=inputs.device),
                    past_key_values=cache, use_cache=True, logits_to_keep=0)
                torch.cuda.synchronize()
                atlas = [self.execution.routes[i].reshape(-1, 8).cpu().tolist() for i in range(48)]
                del output, cache
            finally:
                self.execution.probing = False
            return atlas

        def cycle_level(self, committed_offset):
            """Measured packed/adaptive streams: this cycle's reference routes, packed level by level."""
            iteration = len(self.rows)
            if self.level is None or self.level[0] != iteration:
                source_row = self.reference['cycles'][iteration]
                if source_row['committed_offset'] != committed_offset:
                    raise RuntimeError('prefix differs from independent route preparation')
                self.level = iteration, level_selector(source_row['atlas'] if self.predicted is None
                                                       else self.predicted[iteration])
            return self.level[1]

        def stop_after_level(self, pool, *, committed_offset):
            stop, ids = self.cycle_level(committed_offset).stop(pool.candidates)
            self.stopped = (pool.candidates, ids) if stop else None
            return stop

        def select_by_level(self, pool, fingerprint, committed_offset):
            if self.reference is None:
                if self.method != 'specfit_adaptive':
                    raise RuntimeError('packed SpecFit reuses the SpecFit reference')
                # Probe the full pool once, then replay the stopping rule on its level prefixes.
                atlas = self.probe(pool)
                level = level_selector(atlas if accuracy is None else
                                       predicted_routes(atlas, accuracy, (*self.key, len(self.rows))))
                for depth in range(1, pool.max_depth+1):
                    record = candidate_pool_prefix(pool, depth)
                    stop, ids = level.stop(record.candidates)
                    if stop:
                        break
                return ids, record, [layer[:record.raw_candidate_count+1] for layer in atlas]
            source_row = self.reference['cycles'][len(self.rows)]
            if source_row['pool']['fingerprint'] != fingerprint or source_row['committed_offset'] != committed_offset:
                raise RuntimeError('prefix/pool differs from independent route preparation')
            if self.stopped is not None and self.stopped[0] == pool.candidates:
                ids = self.stopped[1]
            else:
                ids = self.cycle_level(committed_offset).select(pool.candidates)
            self.stopped = None
            return ids, pool, None

        def select_raw_ids(self, pool, *, draft_node_budget, committed_offset=0):
            fingerprint = candidate_pool_fingerprint(pool)
            view = SimpleNamespace(nodes=pool.candidates, fingerprint=fingerprint, seed_token_id=pool.seed_token_id)
            atlas, record = None, pool
            if self.method in LEVEL_METHODS or reference_method(self.method) == 'specfit_adaptive':
                ids, record, atlas = self.select_by_level(pool, fingerprint, committed_offset)
            else:
                if reference_method(self.method) == 'specfit':
                    if self.reference is None:
                        atlas = self.probe(pool)
                    else:
                        source_row = self.reference['cycles'][len(self.rows)]
                        if source_row['pool']['fingerprint'] != fingerprint or source_row['committed_offset'] != committed_offset:
                            raise RuntimeError('prefix/pool differs from independent route preparation')
                        atlas = source_row['atlas']
                ids = choose(view, self.method, model, atlas)
            self.rows.append(dict(pool_object=record, iteration=len(self.rows), committed_offset=committed_offset,
                                 selected_raw_ids=list(ids), atlas=atlas))
            return ids

        def begin_cycle(self, **kw): pass
        def abort_cycle(self): pass
        def finish_generation(self, **kw): pass

        def end_cycle(self, *, committed_token_ids, **kw):
            self.rows[-1]['committed_tokens'] = committed_token_ids.flatten().cpu().tolist()

        def materialize(self):
            result = []
            for row in self.rows:
                value = {k: v for k, v in row.items() if k not in ('pool_object', 'atlas')}
                value['pool'] = serialize_candidate_pool(row['pool_object'])
                if self.reference is None and reference_method(self.method) in ('specfit', 'specfit_adaptive'):
                    value['atlas'] = row['atlas']
                result.append(value)
            return result

    def question(dataset, qid, method, reference, ordinal):
        nonlocal request_start
        gc.enable(); gc.collect(); gc.disable()
        request_start = time.perf_counter()
        event(phase='request_start', dataset=dataset, question_id=qid, method=method, reference=reference is None)
        work = prepared[dataset, qid]
        ids = work['ids'].to('cuda:0')
        attention = torch.ones_like(ids)
        before = cache_boundary(loaded)
        loaded.set_expert_execution_policy(ExpertExecutionPolicy.EXPERT_ID_ASCENDING)
        execution = Execution(method)
        selector = Selector(method, execution, reference, (dataset, qid))
        chain = NativeChainRecorder(drafter, lambda: execution.last_cache) if method in CHAIN_METHODS else None
        prefill_end = None
        def prefill_done():
            nonlocal prefill_end
            torch.cuda.synchronize()
            prefill_end = time.perf_counter_ns()
            loaded.set_expert_execution_policy(ExpertExecutionPolicy.RESIDENT_FIRST)
            execution.decode = True
        total = TOTAL_TOKENS[dataset]
        context = _experimental_2048_token_budget() if total == 2048 else nullcontext()
        with torch.inference_mode(), context, execution.observe():
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            common = dict(attention_mask=attention, max_new_tokens=total-ids.shape[1],
                          stop_on_eos=True, on_prefill_complete=prefill_done)
            if reference_method(method) == 'ar':
                result = target_greedy_generate(loaded.model, ids, **common)
            elif chain is not None:
                result = GreedyChainSpeculator(loaded.model, chain).generate(ids, **common, draft_length=3, trace=chain)
            else:
                # Measured adaptive streams stop drafting per level; its reference drafts the full pool.
                level_stop = (dict(stop_after_level=selector.stop_after_level)
                              if reference_method(method) == 'specfit_adaptive' and reference is not None else {})
                result = ShallowTreeSpeculator(loaded.model, Eagle3TreeDrafter(drafter))._generate(ids, **common,
                    top_k=2, max_depth=5, draft_node_budget=18, audit_commits=reference is None,
                    collect_cycle_metrics=True, selector=ExplicitRawIdTreeSelector(selector),
                    trace=selector, _on_boundary=selector.at_boundary, **level_stop)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
        proofs = execution.materialize()
        tokens = result.tokens.flatten().cpu().tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        after = cache_boundary(loaded)
        cycles, metrics = chain.materialize() if chain is not None else (
            selector.materialize(), [asdict(m) for m in result.tree_cycle_metrics])
        row = dict(protocol=PROTOCOL, status='complete', dataset=dataset, question_id=qid, source_id=work['source_id'],
            method=method, repetition=-1 if reference is None else frozen['worker'], ordinal=ordinal,
            total_tokens=total, input_tokens=ids.shape[1], input_ids_sha256=work['input_ids_sha256'],
            rendered_prompt=work['rendered_prompt'], output_token_ids=tokens, output_tokens=len(tokens), output_text=text,
            stop_reason='eos' if result.stopped_on_eos else 'length', stop_token_id=result.stop_token_id,
            request_ns=end-start, prefill_ns=prefill_end-start, decode_ns=end-prefill_end,
            pre_cache_sha256=before['sha256'], post_cache_sha256=after['sha256'],
            pre_cache_occupied=before['occupied'], post_cache_occupied=after['occupied'],
            pre_cache=before['layers'], post_cache=after['layers'],
            cache_delta={k: after['counters'][k]-before['counters'][k] for k in COUNTERS},
            discard_calls=execution.discard_calls, discard_ns=execution.discard_ns,
            cycles=cycles, cycle_metrics=metrics,
            query_proofs=proofs, commit_audits=[dict(**asdict(a), passed=a.passed) for a in result.commit_audits],
            reference_sha256=canonical_json_sha256(reference) if reference else None,
            quality=numeric_grade(dataset, text, work['evaluation']),
            max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        try:
            verify_record(row, reference, model, total, accuracy, revised)
            budget()
        except BaseException:
            row['status'] = 'failed'
            write_json(args.directory/f'failed-{dataset}-{method}-q{qid}.json.gz', row)
            raise
        request_start = None
        return row

    try:
        loaded = load_offloaded_qwen3(m1._TARGET, device=torch.device('cuda:0'), slots_per_layer=frozen['slots_per_layer'],
            packed_checkpoint_root=m1._PACKED, packed_expert_backing='registered_mmap', verify_packed_hashes=True,
            reduction_mode=ReductionMode.CANONICAL, expert_execution_policy=ExpertExecutionPolicy.EXPERT_ID_ASCENDING,
            h2d_prefetch=False)
        numeric = apply_numeric_mode('math_bf16_full_reduction')
        m1._require_m1_eos_ids(loaded.model)
        index['environment'].update(load_timings=asdict(loaded.load_timings), numeric_settings=dict(numeric.actual_settings))
        drafter = Eagle3ChainDrafter.from_pretrained(m1._DRAFTER, target_root=m1._TARGET,
            device=torch.device('cuda:0'), use_torch_compile=False, expected_target_repo_id=source['target']['repo_id'])
        with _experimental_2048_token_budget():
            warmup_eagle3_tree_drafter(drafter, Eagle3TreeDrafter(drafter), max_sequence_length=2042,
                max_depth=5, top_k=2, draft_node_budget=18, sequence_anchors=(64, 192, 320, 506, 1024, 1536, 2042))
        index['prepare_seconds'] = time.perf_counter()-began
        event(phase='models_ready', load_seconds=loaded.load_timings.total_seconds)
        for dataset in frozen['dataset_order']:
            references = {}
            for method in frozen['reference_methods']:
                reused = reused_stream(args.directory, frozen, dataset, method, True)
                if reused:
                    index['references'].extend(reused)
                    for qid, descriptor in zip(frozen['datasets'][dataset]['question_ids'], reused, strict=True):
                        references[method, qid] = descriptor
                    event(phase='reference_stream_reused', dataset=dataset, method=method, requests=len(reused))
                    continue
                loaded.cold_reset_expert_caches()
                torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                previous = None
                for ordinal, qid in enumerate(frozen['datasets'][dataset]['question_ids']):
                    row = question(dataset, qid, method, None, ordinal)
                    if previous is not None and row['pre_cache_sha256'] != previous:
                        raise ValueError('reference cache continuity changed')
                    previous = row['post_cache_sha256']
                    descriptor = write_json(args.directory/f'reference-{dataset}-{method}-q{qid}.json.gz', row)
                    index['references'].append(descriptor)
                    references[method, qid] = descriptor
                    event(phase='reference_complete', dataset=dataset, method=method, question_id=qid,
                          output_tokens=row['output_tokens'], stop_reason=row['stop_reason'])
            for method in frozen['method_order']:
                reused = reused_stream(args.directory, frozen, dataset, method, False)
                if reused:
                    index['questions'].extend(reused)
                    event(phase='measurement_stream_reused', dataset=dataset, method=method, requests=len(reused))
                    continue
                loaded.cold_reset_expert_caches()
                torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                previous = None
                for ordinal, qid in enumerate(frozen['datasets'][dataset]['question_ids']):
                    d = references[reference_method(method), qid]
                    reference = read_json(args.directory/d['basename'], d)
                    row = question(dataset, qid, method, reference, ordinal)
                    if previous is not None and row['pre_cache_sha256'] != previous:
                        raise ValueError('measurement cache continuity changed')
                    previous = row['post_cache_sha256']
                    descriptor = write_json(args.directory/f'measure-{dataset}-{method}-q{qid}.json.gz', row)
                    index['questions'].append(descriptor)
                    event(phase='measurement_complete', dataset=dataset, method=method, question_id=qid,
                        output_tokens=row['output_tokens'], decode_tps=(row['output_tokens']-1)*1e9/row['decode_ns'],
                        stop_reason=row['stop_reason'])
            write_json(args.directory/f'{dataset}-complete.json', dict(dataset=dataset,
                references=[d for d in index['references'] if d['basename'].startswith('reference-'+dataset+'-')],
                measurements=[d for d in index['questions'] if d['basename'].startswith('measure-'+dataset+'-')]))
        index.update(status='complete', elapsed_seconds=time.perf_counter()-began, finished_ns=time.time_ns())
        write_json(args.directory/'result.json', index)
        report = validate(args.directory, persist=True)
        event(phase='verified', references=report['references'], measured=report['measured_requests'])
    except BaseException as error:
        index.update(status='failed', error=repr(error), elapsed_seconds=time.perf_counter()-began)
        write_json(args.directory/'failure.json', index)
        raise
    finally:
        if numeric is not None:
            numeric.restore()
        if loaded is not None:
            loaded.close()
        if gc_enabled:
            gc.enable()


if __name__ == '__main__':
    args = arguments()
    if args.command == 'freeze': freeze(args)
    elif args.command == 'run': run(args)
    else: print(json.dumps(validate(args.directory), ensure_ascii=False, indent=2))
