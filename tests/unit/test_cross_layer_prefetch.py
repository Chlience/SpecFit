from copy import deepcopy
import unittest

from specfit.cross_layer_prefetch import replay_events, schedule, predictions_for


def empty_state():
    return dict(layers=[dict(layer_id=0, capacity=1, lru_sequence=0, submission_sequence=0,
        counters=dict(h2d_submissions=0, resident_hits=0, inflight_joins=0, evictions=0, pending_use_guards=0),
        slots=[dict(slot_id=0, expert_id=None, generation=0, active_leases=0,
                    last_used_sequence=0, ready=False, pending_use_guards=0)])])


def events():
    return [["h2d_submit", 0, 3, 0, 1, 0, None],
            ["consume_wait", 0, 3, 0, 1, None, None],
            ["release", 0, 3, 0, 1, None, None]]


class PrefetchEventAuditTest(unittest.TestCase):
    def test_replay_preserves_input_and_reconstructs_touch_counters(self):
        pre = empty_state()
        original = deepcopy(pre)
        post = replay_events(pre, events(), {(0, 3)}, "predicted_cross_layer")
        self.assertEqual(pre, original)
        self.assertEqual(post["layers"][0]["lru_sequence"], 2)
        self.assertEqual(post["layers"][0]["counters"]["h2d_submissions"], 1)
        self.assertEqual(post["layers"][0]["slots"][0]["expert_id"], 3)

    def test_wrong_prediction_may_release_without_computation(self):
        rows = events()
        rows.pop(1)
        post = replay_events(empty_state(), rows, set(), "wrong_cross_layer")
        self.assertEqual(post["layers"][0]["lru_sequence"], 1)

    def test_stale_generation_duplicate_compute_missing_release_and_sequence_rejected(self):
        bad = []
        rows = events(); rows[1][4] = 2; bad.append(rows)
        rows = events(); rows.insert(2, rows[1].copy()); bad.append(rows)
        bad.append(events()[:-1])
        rows = events(); rows[0][5] = 1; bad.append(rows)
        for rows in bad:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replay_events(empty_state(), rows, {(0, 3)}, "predicted_cross_layer")

    def test_non_router_computation_rejected(self):
        with self.assertRaises(ValueError):
            replay_events(empty_state(), events(), {(0, 4)}, "predicted_cross_layer")

    def test_protected_slot_cannot_be_evicted(self):
        rows = [events()[0], ["evict", 0, 4, 0, 2, None, 3]]
        with self.assertRaisesRegex(ValueError, "leased victim"):
            replay_events(empty_state(), rows, set(), "predicted_cross_layer")

    def test_unpaired_eviction_rejected(self):
        rows = events() + [["evict", 0, 4, 0, 2, None, 3]]
        with self.assertRaises(ValueError):
            replay_events(empty_state(), rows, {(0, 3)}, "predicted_cross_layer")

    def test_accurate_predictions_must_preserve_unconsumed_residents(self):
        pre = empty_state()
        layer = pre["layers"][0]
        template = layer["slots"][0]
        layer.update(capacity=3, submission_sequence=3, lru_sequence=3,
            slots=[dict(template, slot_id=sid, expert_id=expert, generation=1,
                        ready=True, last_used_sequence=sid + 1)
                   for sid, expert in enumerate((2, 1, 3))])
        layer["counters"]["h2d_submissions"] = 3
        rows = [["evict", 0, 0, 0, 2, None, 2], ["h2d_submit", 0, 0, 0, 2, 3, None],
                ["consume_wait", 0, 0, 0, 2, None, None], ["release", 0, 0, 0, 2, None, None],
                ["evict", 0, 2, 0, 3, None, 0], ["h2d_submit", 0, 2, 0, 3, 4, None],
                ["consume_wait", 0, 2, 0, 3, None, None], ["release", 0, 2, 0, 3, None, None]]
        for variant in ("predicted_same_layer", "predicted_cross_layer"):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "unconsumed actual"):
                replay_events(pre, rows, {(0, 0), (0, 2)}, variant)
        # Imperfect predictions can incur extra transfers while remaining safe.
        post = replay_events(pre, rows, {(0, 0), (0, 2)}, "wrong_cross_layer")
        self.assertEqual(post["layers"][0]["counters"]["h2d_submissions"], 5)

    def test_frozen_schedule_and_disjoint_wrong_predictions(self):
        rows = schedule("q416-c37")
        self.assertEqual(rows, schedule("q416-c37"))
        self.assertEqual(len(rows), 52)
        self.assertEqual(sum(r["phase"] == "measurement" for r in rows), 30)
        reference = dict(routes=dict(rows=[dict(layer_id=0, expert_ids=[0, 2, 4])]))
        self.assertFalse(set(predictions_for(reference, "wrong_cross_layer")[0]) & {0, 2, 4})
        self.assertEqual(predictions_for(reference, "ready_demand"), {})
