from copy import deepcopy
from itertools import combinations
from math import exp
from pathlib import Path
import sys
import tempfile
import unittest

from specfit.experiment import canonical_json_sha256
from specfit.r6a_experiment import compute_r6a_pool_fingerprint
from specfit.serial_replay import (COUNTERS, PROTOCOL, choose, numeric_answer,
    pool_view, route_atlas, validate_capture, validate_stream)
from tests.unit.r6a_fixture import R6AFixture
from tests.unit.test_selection_refinement import SelectionRefinementTest


class SerialReplayTest(unittest.TestCase):
    def pool(self, count):
        value = R6AFixture.pool()
        value.update(candidates=value["candidates"][:count], raw_candidate_count=count,
                     max_depth=(count + 2) // 4)
        value["fingerprint"] = compute_r6a_pool_fingerprint(value)
        return value

    def atlas(self, count):
        return [[[(raw * 11 + lid * 3 + e) % 128 for e in range(8)]
                 for raw in range(count + 1)] for lid in range(48)]

    def test_tail_selection_matches_independent_exhaustive_objective(self):
        model = SelectionRefinementTest.model()
        for count in (2, 6, 10, 14, 18):
            pool = pool_view(self.pool(count))
            atlas = self.atlas(count)
            candidates = {n.raw_id: n for n in pool.nodes}
            scores = []
            for k in range(min(7, count) + 1):
                for selected in combinations(candidates, k):
                    if any(candidates[i].parent_raw_id not in (0, *selected) for i in selected):
                        continue
                    union = {(lid, expert) for raw in (0, *selected) for lid in range(48)
                             for expert in atlas[lid][raw]}
                    utility = 1 + sum(exp(candidates[i].cumulative_log_probability) for i in selected)
                    elapsed = model.size_intercept_ns[k] + model.expert_ns * len(union)
                    scores.append(((-utility / elapsed, -utility, len(union), k, selected), selected))
            self.assertEqual(choose(pool, "expert_fast_dynamic", model, atlas), min(scores)[1])
            selected = choose(pool, "probability_dynamic", model)
            self.assertLessEqual(len(selected), min(7, count))
            self.assertTrue(all(candidates[i].parent_raw_id in (0, *selected) for i in selected))

    def test_atlas_rejects_missing_layers_and_duplicate_experts(self):
        pool = pool_view(self.pool(2))
        for atlas in (self.atlas(2)[:-1], [[list(range(7)) + [0]] * 3] * 48):
            with self.assertRaises(ValueError):
                route_atlas(pool, atlas)
        value = self.pool(2)
        value["seed_token_id"] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            pool_view(value)

    def test_capture_binds_selection_committed_prefix_and_audit(self):
        pool = self.pool(2)
        model = SelectionRefinementTest.model()
        atlas = self.atlas(2)
        selected = choose(pool_view(pool), "expert_fast_dynamic", model, atlas)
        row = dict(protocol=PROTOCOL, method="expert_fast_dynamic",
            cycles=[dict(iteration=0, committed_offset=0, pool=pool, atlas=atlas,
                         selected_raw_ids=list(selected), committed_tokens=[42])],
            output_token_ids=[42, 151645], commit_audits=[dict(passed=True)])
        self.assertEqual(validate_capture(row, model), 1)
        for field, value in (("committed_offset", 1), ("selected_raw_ids", [2]),
                             ("committed_tokens", [999])):
            changed = deepcopy(row)
            changed["cycles"][0][field] = value
            with self.assertRaises(ValueError):
                validate_capture(changed, model)
        changed = deepcopy(row)
        changed["commit_audits"] = []
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_capture(changed, model)

    def rows(self):
        empty = [dict(slots=[dict(expert=None)], stats=dict.fromkeys(COUNTERS, 0))]
        warm = [dict(slots=[dict(expert=3)], stats=dict.fromkeys(COUNTERS, 1))]
        later = [dict(slots=[dict(expert=4)], stats=dict.fromkeys(COUNTERS, 2))]
        result = []
        for i, (before, after) in enumerate(((empty, warm), (warm, later))):
            tokens = [7, 151645] if i == 0 else [8, 9]
            result.append(dict(protocol=PROTOCOL, method="target_only", repetition=0,
                ordinal=i, question_id=401+i, status="complete", input_tokens=510,
                output_token_ids=tokens, output_tokens=2, stop_reason="length" if i else "eos",
                stop_token_id=None if i else 151645, prefill_ns=1, request_ns=3, decode_ns=2,
                pre_cache=before, post_cache=after, pre_cache_occupied=i, post_cache_occupied=1,
                pre_cache_sha256=canonical_json_sha256(before), post_cache_sha256=canonical_json_sha256(after),
                cache_delta=dict.fromkeys(COUNTERS, 1), all_commits_passed=True, reference_match=True,
                max_gpu_reserved_bytes=1, max_rss_bytes=1, answer_correct=not i,
                predicted_answer="7" if not i else None))
        return result

    def check_stream(self, rows):
        return validate_stream(rows, question_ids=[401, 402], method="target_only", repetition=0)

    def test_ordered_stream_keeps_truncated_question_and_checks_cache_continuity(self):
        summary = self.check_stream(self.rows())
        self.assertEqual(summary, dict(questions=2, output_tokens=4, decode_tokens=2,
            request_ns=6, decode_ns=4, truncated=1, answer_correct=1, answer_unparsed=1))
        for field, value in (("pre_cache_sha256", "reset"), ("input_tokens", 511),
                             ("output_tokens", 3), ("stop_reason", "eos"),
                             ("decode_ns", 3), ("reference_match", False),
                             ("max_gpu_reserved_bytes", 39 * 1024**3)):
            rows = self.rows()
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check_stream(rows)
        with self.assertRaisesRegex(ValueError, "every question"):
            self.check_stream(self.rows()[:1])
        rows = self.rows()
        rows[1]["post_cache"][0]["stats"]["evictions"] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            self.check_stream(rows)

    def test_answer_parser_does_not_treat_intermediate_number_as_answer(self):
        self.assertEqual(numeric_answer("Calculation 2, then #### 1,025.0"), "1025")
        self.assertEqual(numeric_answer(r"Thus \boxed{25.00}"), "25")
        self.assertEqual(numeric_answer("The final answer $25"), "25")
        self.assertEqual(numeric_answer("### ✅ Final Answer: **$1,110**"), "1.11E+3")
        self.assertEqual(numeric_answer("The answer is __25.00__"), "25")
        self.assertEqual(numeric_answer(r"Thus \boxed{\$1,110}"), "1.11E+3")
        self.assertIsNone(numeric_answer("There are 12 initially, then add 3."))
        self.assertIsNone(numeric_answer(r"\boxed{25}", reference=True))

    def test_artifact_is_exclusive_and_byte_checked(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import run_qwen3_serial_replay as runner
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json.gz"
            descriptor = runner.write_json(path, {"tokens": [1, 2]})
            self.assertEqual(runner.read_json(path, descriptor), {"tokens": [1, 2]})
            with self.assertRaises(FileExistsError):
                runner.write_json(path, {"replacement": True})
            with path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                runner.read_json(path, descriptor)
            with self.assertRaises(FileNotFoundError):
                runner.validate(Path(directory))


if __name__ == "__main__":
    unittest.main()
