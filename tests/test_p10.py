import unittest

import numpy as np

from research_impl.methods.p10 import active_episodes, correctness_checks, enumerate_reappearing_demands


class P10Tests(unittest.TestCase):
    def test_toys(self):
        self.assertTrue(correctness_checks()["passed"])

    def test_episode_scores_are_signed_free_k2_sums(self):
        s = np.zeros((24, 2), dtype=np.float64)
        e = np.zeros_like(s)
        s[[0, 1, 4, 5], 0] = 1
        e[[0, 1, 4, 5], 0] = [1, 2, 3, 4]
        demands = enumerate_reappearing_demands(s, e, static_count=1)
        self.assertEqual(len(demands), 2)
        self.assertEqual([row["episode_score"] for row in demands], [7.0, 3.0])
        self.assertEqual(demands[0]["reappearance_sample"], 4)

    def test_short_gap_is_one_episode(self):
        episodes = active_episodes(np.asarray([1, 1, 0, 1, 1], dtype=np.float64))
        self.assertEqual(len(episodes), 1)


if __name__ == "__main__":
    unittest.main()
