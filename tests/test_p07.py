import unittest

import numpy as np
import torch

from research_impl.methods.p07 import common_id_diagnostics, round_state, round_target, score_maps, select_round


class FakeAdapter:
    def __init__(self, static_rows, dynamic_rows):
        self.static_rows = np.asarray(static_rows, dtype=np.int64)
        self.dynamic_rows = np.asarray(dynamic_rows, dtype=np.int64)

    @property
    def static_count(self):
        return len(self.static_rows)

    @property
    def dynamic_count(self):
        return len(self.dynamic_rows)

    @property
    def total_count(self):
        return self.static_count + self.dynamic_count

    def gather(self, static_indices, dynamic_indices):
        return FakeAdapter(self.static_rows[static_indices], self.dynamic_rows[dynamic_indices])


class P07Tests(unittest.TestCase):
    def test_round_targets_end_at_frozen_group_quotas(self):
        self.assertEqual([round_target(185033, 92516, j) for j in range(1, 5)], [161904, 138775, 115646, 92516])
        self.assertEqual([round_target(63862, 31931, j) for j in range(1, 5)], [55880, 47897, 39914, 31931])

    def test_stale_four_rounds_equal_one_shot(self):
        original = FakeAdapter(np.arange(12), np.arange(100, 108))
        original_scores = np.asarray([0.2, 0.7, 0.1, 0.9, 0.6, 0.3, 0.8, 0.4, 0.5, 0.0, 1.0, 0.05, 0.5, 0.1, 0.8, 0.2, 0.6, 0.3, 0.9, 0.4])
        stale_static, stale_dynamic = score_maps(original, original_scores)
        current = original
        for round_index in range(1, 5):
            # Fresh diagnostics deliberately change, but stale scores decide.
            fresh = np.linspace(1.0, 0.0, current.total_count)
            cache = {"e_it": torch.from_numpy(fresh[None, :])}
            selection = select_round(
                current,
                cache,
                round_target(12, 6, round_index),
                round_target(8, 4, round_index),
                stale_static,
                stale_dynamic,
            )
            current = current.gather(selection.static_indices, selection.dynamic_indices)
        expected_static = np.lexsort((original.static_rows, -original_scores[:12]))[:6]
        expected_dynamic = np.lexsort((original.dynamic_rows, -original_scores[12:]))[:4]
        np.testing.assert_array_equal(np.sort(current.static_rows), np.sort(original.static_rows[expected_static]))
        np.testing.assert_array_equal(np.sort(current.dynamic_rows), np.sort(original.dynamic_rows[expected_dynamic]))

    def test_newly_exposed_ray_count_uses_positive_hit_delta(self):
        previous_adapter = FakeAdapter([0, 1, 2], [10])
        previous_scores = np.asarray([1.0, 2.0, 3.0, 4.0])
        previous_hits = np.asarray([[4, 2, 1, 3], [1, 1, 1, 1]])
        previous = round_state(previous_adapter, previous_scores, previous_hits)
        current = FakeAdapter([1, 2], [10])
        current_scores = np.asarray([2.2, 2.8, 4.2])
        current_hits = np.asarray([[5, 0, 4], [1, 3, 0]])
        diagnostics = common_id_diagnostics(previous, current, current_scores, current_hits)
        self.assertEqual(diagnostics["common_ids"], 3)
        self.assertEqual(diagnostics["newly_exposed_ray_count"], 6)
        self.assertEqual(diagnostics["newly_hidden_ray_count"], 2)


if __name__ == "__main__":
    unittest.main()
