from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from specfit.experiment import canonical_json_sha256
from specfit.main_comparison import (PILOT_IDS, PROMPT_BUDGETS, SMOKE_IDS, TOTAL_TOKENS, LevelSpecFit,
    RevisedCostModel, RevisedLevelSpecFit, choose, exhaustive_revised_choice, exhaustive_specfit_choice, extract_python,
    load_prompt_budget, numeric_grade, predicted_routes, reference_method, verify_adaptive_depth, verify_record)
from specfit.selection_methods import probability_ids, select_nodes
from specfit.serial_replay import pool_view, route_atlas
from tests.unit import test_serial_replay as serial_fixture
from tests.unit import test_selection_refinement as selection_fixture


class MainComparisonTest(unittest.TestCase):
    def test_eagle_cache_switch_preserves_fixed_probability_choice_and_specfit_is_original(self):
        fixture = serial_fixture.SerialReplayTest()
        model = selection_fixture.SelectionRefinementTest.model()
        for count in (2, 6, 10, 14, 18):
            pool = pool_view(fixture.pool(count))
            atlas = fixture.atlas(count)
            expected = probability_ids(pool, min(7, count))
            for method in ('eagle_cache', 'eagle_no_cache'):
                self.assertEqual(choose(pool, method, model), expected)
            for method in ('specfit', 'specfit_no_cache'):
                self.assertEqual(choose(pool, method, model, atlas),
                                 exhaustive_specfit_choice(pool, model, atlas))
            if count == 18:
                self.assertEqual(exhaustive_specfit_choice(pool, model, atlas),
                                 select_nodes(pool, route_atlas(pool, atlas), 'expert_cost_dynamic', model))

    def test_level_specfit_matches_exhaustive_and_adaptive_depth_is_rechecked(self):
        fixture = serial_fixture.SerialReplayTest()
        model = selection_fixture.SelectionRefinementTest.model()
        # Fixture routes leave level 3 empty; routes shared with the seed leave level 4 empty.
        shared = [[[(lid * 3 + e) % 128 for e in range(8)] for _ in range(19)] for lid in range(48)]
        for atlas, stop in ((fixture.atlas(18), 3), (shared, 4)):
            level, empties = LevelSpecFit(model, atlas), []
            for depth in range(1, 6):
                count = 4 * depth - 2
                pool, rows = pool_view(fixture.pool(count)), [layer[:count + 1] for layer in atlas]
                ids = exhaustive_specfit_choice(pool, model, rows)
                empties.append(all(pool.nodes[raw - 1].depth != depth for raw in ids))
                with self.subTest(stop=stop, depth=depth):
                    # One instance packs each level once; a fresh instance packs the prefix at once.
                    self.assertEqual(level.stop(pool.nodes), (empties[-1], ids))
                    self.assertEqual(LevelSpecFit(model, atlas).select(pool.nodes), ids)
                    if depth == stop:
                        self.assertEqual(verify_adaptive_depth(pool, model, rows, 5), ids)
                    else:
                        # Drafting past an empty level and stopping before one are both rejected.
                        with self.assertRaises(ValueError): verify_adaptive_depth(pool, model, rows, 5)
                    if depth < stop:
                        # A generation tail ends drafting without an empty level.
                        self.assertEqual(verify_adaptive_depth(pool, model, rows, depth), ids)
                    with self.assertRaises(ValueError): verify_adaptive_depth(pool, model, rows, depth - 1)
                    with self.assertRaises(ValueError):
                        verify_adaptive_depth(pool, model, [layer[:count] for layer in atlas], 5)
            self.assertEqual(empties.index(True) + 1, stop)
        pool = pool_view(fixture.pool(10))
        with self.assertRaises(ValueError): LevelSpecFit(model, fixture.atlas(6)).select(pool.nodes)
        repeated = deepcopy(fixture.atlas(10)); repeated[7][9][3] = repeated[7][9][4]
        with self.assertRaises(ValueError): LevelSpecFit(model, repeated).select(pool.nodes)
        with self.assertRaises(ValueError): LevelSpecFit(model, fixture.atlas(10)).select(pool.nodes[:9])

    def test_revised_level_specfit_matches_exhaustive_and_charges_the_next_level(self):
        fixture = serial_fixture.SerialReplayTest()
        model = selection_fixture.SelectionRefinementTest.model()
        shared = [[[(lid * 3 + e) % 128 for e in range(8)] for _ in range(19)] for lid in range(48)]
        draft = (7e6, 19e6, 31e6, 43e6, 55e6)
        cheap = RevisedCostModel(model.size_intercept_ns, model.expert_ns, (1.0, 2.0, 3.0, 4.0, 5.0))
        costly = RevisedCostModel(model.size_intercept_ns, model.expert_ns, draft)
        for revised in (cheap, costly):
            for atlas in (fixture.atlas(18), shared):
                level = RevisedLevelSpecFit(revised, atlas)
                for depth in range(1, 6):
                    count = 4 * depth - 2
                    pool, rows = pool_view(fixture.pool(count)), [layer[:count + 1] for layer in atlas]
                    ids = exhaustive_revised_choice(pool, revised, rows)
                    with self.subTest(draft=revised.draft_ns, depth=depth):
                        stop, picked = level.stop(pool.nodes)
                        self.assertEqual(picked, ids)
                        self.assertEqual(RevisedLevelSpecFit(revised, atlas).select(pool.nodes), ids)
                        self.assertEqual(stop, all(pool.nodes[raw - 1].depth != depth for raw in ids))
                        if stop:
                            self.assertEqual(verify_adaptive_depth(pool, model, rows, 5, revised), ids)
                            break
                        with self.assertRaises(ValueError): verify_adaptive_depth(pool, model, rows, 5, revised)
        # A 12 ms charge for the next level never adds depth-d nodes relative to a negligible one.
        pool, rows = pool_view(fixture.pool(10)), [layer[:11] for layer in fixture.atlas(18)]
        deep = lambda ids: sum(pool.nodes[raw - 1].depth == 3 for raw in ids)
        self.assertLessEqual(deep(exhaustive_revised_choice(pool, costly, rows)),
                             deep(exhaustive_revised_choice(pool, cheap, rows)))
        self.assertEqual(RevisedCostModel.from_dict(dict(size_intercept_ns=list(model.size_intercept_ns),
                         expert_ns=model.expert_ns, draft_ns=list(draft))), costly)
        with self.assertRaises(ValueError): RevisedCostModel(model.size_intercept_ns, model.expert_ns, draft[::-1])
        with self.assertRaises(ValueError): RevisedCostModel(model.size_intercept_ns[:7], model.expert_ns, draft)
        with self.assertRaisesRegex(ValueError, 'adaptive SpecFit'):
            verify_record(dict(method='eagle_cache'), None, model, 512, None, costly)

    def test_predicted_routes_are_deterministic_prefix_stable_and_keep_the_given_fraction(self):
        fixture = serial_fixture.SerialReplayTest()
        model = selection_fixture.SelectionRefinementTest.model()
        atlas, key = fixture.atlas(18), ('gsm8k', 402, 3)
        predicted = predicted_routes(atlas, 0.8, key)
        self.assertEqual(predicted, predicted_routes(atlas, 0.8, key))
        self.assertNotEqual(predicted, predicted_routes(atlas, 0.8, ('gsm8k', 402, 4)))
        self.assertEqual(predicted_routes(atlas, 1.0, key), [[sorted(row) for row in layer] for layer in atlas])
        pairs = [(t, q) for true, guess in zip(atlas, predicted) for t, q in zip(true, guess)]
        self.assertTrue(all(len(set(q)) == 8 and all(0 <= e < 128 for e in q) for _, q in pairs))
        self.assertAlmostEqual(sum(len(set(t) & set(q)) for t, q in pairs) / (8 * len(pairs)), 0.8, delta=0.03)
        # A depth<=d prefix predicts as the prefix of the full pool, so the recheck sees the routes the run selected with.
        for count in (2, 6, 10, 14):
            self.assertEqual(predicted_routes([layer[:count + 1] for layer in atlas], 0.8, key),
                             [layer[:count + 1] for layer in predicted])
        level = LevelSpecFit(model, predicted)
        for depth in range(1, 6):
            pool = pool_view(fixture.pool(4 * depth - 2))
            stop, ids = level.stop(pool.nodes)
            if stop: break
        stored = [layer[:len(pool.nodes) + 1] for layer in atlas]
        self.assertEqual(verify_adaptive_depth(pool, model, predicted_routes(stored, 0.8, key), 5), ids)
        with self.assertRaises(ValueError): predicted_routes(atlas, 0.0, key)
        with self.assertRaisesRegex(ValueError, 'adaptive SpecFit'):
            verify_record(dict(method='eagle_cache'), None, model, 512, 0.9)

    def test_method_names_do_not_silently_select_a_different_baseline(self):
        self.assertEqual(reference_method('eagle_no_cache'), 'eagle_cache')
        self.assertEqual(reference_method('specfit_packed'), 'specfit')
        self.assertEqual(reference_method('specfit_adaptive'), 'specfit_adaptive')
        self.assertEqual(reference_method('specfit_adaptive_no_cache'), 'specfit_adaptive')
        self.assertEqual(reference_method('specfit_no_cache'), 'specfit')
        self.assertEqual(reference_method('ar_no_cache'), 'ar')
        with self.assertRaises(ValueError): reference_method('specfit_cache')
        with self.assertRaises(ValueError): reference_method('probability_dynamic')
        with self.assertRaises(ValueError): choose(None, 'ar', None)

    def test_scoring_requires_explicit_answer_and_preserves_humaneval_pending_status(self):
        self.assertEqual(numeric_grade('aime25', r'\boxed{030}', dict(answer='30'))['correct'], True)
        for text in ('Intermediate calculation 30.', r'\boxed{30.5}', r'\boxed{1000}'):
            self.assertEqual(numeric_grade('aime25', text, dict(answer='30'))['status'], 'unparsed')
        self.assertEqual(numeric_grade('gsm8k', '#### 1,000', dict(answer='1E+3'))['correct'], True)
        self.assertIsNone(numeric_grade('humaneval', 'def f(): pass', {})['correct'])
        self.assertEqual(numeric_grade('summarization', 'The article reports 42 deaths.', dict(reference='42')),
                         dict(status='not_scored', predicted=None, correct=None))

    def test_prompt_budgets_merge_only_with_one_tokenizer_and_disjoint_datasets(self):
        root = Path(__file__).resolve().parents[2]
        merged = load_prompt_budget(root)
        self.assertEqual(set(merged['datasets']), {'humaneval', 'aime25', 'summarization'})
        self.assertEqual(merged['total_tokens'], TOTAL_TOKENS['summarization'])
        requests = merged['datasets']['summarization']['requests']
        self.assertEqual([r['question_id'] for r in requests], list(range(80)))
        self.assertTrue(all(r['fits'] and r['input_tokens']+r['available_output_tokens'] == 2048 for r in requests))
        # Smoke covers the longest article, so the formal run cannot meet a longer prefill first.
        self.assertIn(max(requests, key=lambda r: r['input_tokens'])['question_id'], SMOKE_IDS['summarization'])
        # The pilot spreads over input length: the rule's ranks reproduce its ids, shortest and longest included.
        ranked = sorted(requests, key=lambda r: (r['input_tokens'], r['question_id']))
        self.assertEqual(sorted(ranked[round(i*79/7)]['question_id'] for i in range(8)), list(PILOT_IDS['summarization']))
        base = json.loads((root/PROMPT_BUDGETS[0]).read_text())
        for other in (dict(base), dict(base, datasets={'summarization': {}}, tokenizer_files={'vocab.json': 'changed'})):
            with tempfile.TemporaryDirectory() as temp:
                for relative, value in zip(PROMPT_BUDGETS, (base, other)):
                    (Path(temp)/relative).parent.mkdir(parents=True, exist_ok=True)
                    (Path(temp)/relative).write_text(json.dumps(value))
                with self.assertRaises(ValueError): load_prompt_budget(Path(temp))

    def test_code_extraction_does_not_repair_ambiguous_answers(self):
        self.assertEqual(extract_python('```python\ndef f():\n    return 1\n```'), 'def f():\n    return 1')
        self.assertEqual(extract_python('def f():\n    return 1'), 'def f():\n    return 1')
        with self.assertRaises(ValueError): extract_python('```python\nx=1\n```\n```python\nx=2\n```')

    def test_record_rejects_replaced_query_proof_bad_counter_and_missing_eos(self):
        row = serial_fixture.SerialReplayTest().rows()[0]
        row.update(method='ar', dataset='gsm8k', total_tokens=512, cycles=[], cycle_metrics=[],
            query_proofs=[dict(finite=True, expert_union=384)]*2, commit_audits=[], discard_calls=0,
            discard_ns=0, input_ids_sha256='inputs')
        model = selection_fixture.SelectionRefinementTest.model()
        verify_record(row, None, model, 512)
        measured = deepcopy(row); measured['reference_sha256'] = canonical_json_sha256(row)
        verify_record(measured, row, model, 512)
        for key, value in (('query_proofs', [dict(finite=True, expert_union=383)]*2),
                           ('cache_delta', dict.fromkeys(row['cache_delta'], 2)), ('stop_token_id', 10)):
            changed = deepcopy(measured); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): verify_record(changed, row, model, 512)

    def test_no_cache_ar_uses_ar_rules_and_must_transfer_every_forward_union(self):
        row = serial_fixture.SerialReplayTest().rows()[0]
        row.update(method='ar', dataset='gsm8k', total_tokens=512, cycles=[], cycle_metrics=[],
            query_proofs=[dict(finite=True, expert_union=384)]*2, commit_audits=[], discard_calls=0,
            discard_ns=0, input_ids_sha256='inputs')
        model = selection_fixture.SelectionRefinementTest.model()
        measured = deepcopy(row)
        measured.update(method='ar_no_cache', discard_calls=2, discard_ns=5, reference_sha256=canonical_json_sha256(row))
        measured['post_cache'][0]['stats']['h2d_submissions'] = row['pre_cache'][0]['stats']['h2d_submissions']+768
        measured['post_cache_sha256'] = canonical_json_sha256(measured['post_cache'])
        measured['cache_delta']['h2d_submissions'] = 768
        verify_record(measured, row, model, 512)
        for key, value in (('discard_calls', 1), ('cycle_metrics', [dict(iteration=0)])):
            changed = deepcopy(measured); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): verify_record(changed, row, model, 512)


if __name__ == '__main__': unittest.main()
