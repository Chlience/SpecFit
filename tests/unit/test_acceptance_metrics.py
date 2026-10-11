from copy import deepcopy
import unittest

from specfit.acceptance_metrics import acceptance_lengths


class AcceptanceLengthTest(unittest.TestCase):
    def tree(self):
        return dict(method="expert_fast_dynamic",output_tokens=4,output_token_ids=[11,22,33,44],
            scheduler=dict(forwards=2),
            cycles=[dict(iteration=0,committed_offset=0,selected_raw_ids=[1,2,3],committed_tokens=[11,22]),
                    dict(iteration=1,committed_offset=2,selected_raw_ids=[],committed_tokens=[33])],
            cycle_metrics=[dict(iteration=0,committed_output_offset=0,selected_draft_nodes=3,
                accepted_draft_nodes=1,verification_raw_ids=[0,1,2,3],accepted_raw_ids=[1]),
                dict(iteration=1,committed_output_offset=2,selected_draft_nodes=0,
                accepted_draft_nodes=0,verification_raw_ids=[0],accepted_raw_ids=[])])

    def test_branch_nodes_and_accepted_path_are_distinct_and_tail_is_excluded(self):
        actual=acceptance_lengths([self.tree()])
        self.assertEqual(actual["verification_rounds"],2)
        self.assertEqual(actual["verification_length_including_root"],dict(total=5,mean=2.5,histogram={"1":1,"4":1}))
        self.assertEqual(actual["accepted_draft_length"]["mean"],.5)
        self.assertEqual(actual["committed_length_including_root"]["mean"],1.5)
        self.assertEqual(actual["accepted_fraction_of_selected_nodes"],1/3)

    def test_ar_counts_actual_decode_forwards_and_has_no_draft_acceptance_rate(self):
        row=dict(method="target_only",output_tokens=4,scheduler=dict(forwards=3),cycles=[],cycle_metrics=[])
        actual=acceptance_lengths([row])
        self.assertEqual(actual["verification_length_including_root"]["histogram"],{"1":3})
        self.assertEqual(actual["accepted_draft_length"]["total"],0)
        self.assertIsNone(actual["accepted_fraction_of_selected_nodes"])
        row.update(output_tokens=1,scheduler=dict(forwards=0))
        self.assertIsNone(acceptance_lengths([row])["verification_length_including_root"]["mean"])

    def test_runtime_tuples_match_json_lists_without_relaxing_path_checks(self):
        row = self.tree()
        expected = acceptance_lengths([row])
        for metric in row["cycle_metrics"]:
            metric["verification_raw_ids"] = tuple(metric["verification_raw_ids"])
            metric["accepted_raw_ids"] = tuple(metric["accepted_raw_ids"])
        self.assertEqual(acceptance_lengths([row]), expected)
        for ids in ((0, 2, 1, 3), (0, 1, 2), (0, 1, 2, 2)):
            with self.subTest(ids=ids):
                changed = deepcopy(row)
                changed["cycle_metrics"][0]["verification_raw_ids"] = ids
                with self.assertRaises(ValueError):
                    acceptance_lengths([changed])

    def test_refuses_inconsistent_counts_commits_and_missing_rounds(self):
        for change in (lambda r:r["cycle_metrics"][0].update(accepted_draft_nodes=2),
                       lambda r:r["cycle_metrics"][0].update(verification_raw_ids=[0,1]),
                       lambda r:r["cycles"][0].update(committed_tokens=[11,99]),
                       lambda r:r["scheduler"].update(forwards=3),
                       lambda r:r.update(output_tokens=6)):
            row=deepcopy(self.tree());change(row)
            with self.assertRaises(ValueError):acceptance_lengths([row])


if __name__ == "__main__":
    unittest.main()
