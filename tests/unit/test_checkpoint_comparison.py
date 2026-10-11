from copy import deepcopy
import unittest

from specfit.experiment import canonical_json_sha256
from specfit.main_comparison import reference_method, verify_record
from tests.unit.test_serial_replay import SerialReplayTest
from tests.unit.test_selection_refinement import SelectionRefinementTest


class CheckpointComparisonTest(unittest.TestCase):
    def row(self, *, shortened=False):
        row = SerialReplayTest().rows()[0]
        prefix = 510 if shortened else 16
        query = [7, 151645] if shortened else [7, 151645, 8, 9]
        greedy = [151645, 4] if shortened else [151645, 8, 9, 4]
        selected = list(range(1, len(query)))
        row.update(method='eagle_chain3_cache', dataset='gsm8k', total_tokens=512,
            input_tokens=prefix, input_ids_sha256='input', commit_audits=[], discard_calls=0, discard_ns=0,
            cycles=[dict(iteration=0, committed_offset=0, selected_raw_ids=selected,
                verification_tokens=query, verifier_tokens=greedy, committed_tokens=[7, 151645],
                base_kv_length=prefix, proposal_prefix_tokens=prefix, kv_length_after=prefix+2,
                observed_kv_length_after=prefix+2)],
            cycle_metrics=[dict(iteration=0, committed_output_offset=0, verification_raw_ids=[0, *selected],
                accepted_raw_ids=[1], selected_draft_nodes=len(selected), accepted_draft_nodes=1)],
            query_proofs=[dict(finite=True, expert_union=1, logits_shape=[1, 1, 151936], greedy_tokens=[7]),
                          dict(finite=True, expert_union=1, logits_shape=[1, len(query), 151936], greedy_tokens=greedy)])
        return row

    def test_native_chain_eos_caps_acceptance_and_tail_caps_proposal(self):
        model = SelectionRefinementTest.model()
        for shortened in (False, True):
            row = self.row(shortened=shortened)
            verify_record(row, None, model, 512)
            measured = deepcopy(row)
            measured['reference_sha256'] = canonical_json_sha256(row)
            verify_record(measured, row, model, 512)

    def test_rejects_wrong_acceptance_query_prefix_and_kv_crop(self):
        model = SelectionRefinementTest.model()
        for key, value in (('verifier_tokens', [0, 8, 9, 4]), ('observed_kv_length_after', 20),
                           ('proposal_prefix_tokens', 15), ('selected_raw_ids', [1, 2]),
                           ('committed_tokens', [7])):
            row = self.row()
            row['cycles'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_record(row, None, model, 512)

    def test_native_cache_switch_checks_transfer_conservation(self):
        self.assertEqual(reference_method('eagle_chain3_no_cache'), 'eagle_chain3_cache')
        row = self.row()
        row.update(method='eagle_chain3_no_cache', discard_calls=2, discard_ns=1)
        row['query_proofs'][0]['expert_union'] = 0
        model = SelectionRefinementTest.model()
        verify_record(row, None, model, 512)
        row['query_proofs'][0]['expert_union'] = 1
        with self.assertRaisesRegex(ValueError, 'H2D'):
            verify_record(row, None, model, 512)


if __name__ == '__main__':
    unittest.main()
