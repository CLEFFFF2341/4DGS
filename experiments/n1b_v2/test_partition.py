import unittest
import numpy as np
import torch
from experiments.n1b.scenes import scene
from experiments.n1b.narrow_benchmark import fixture
from .partition import nodes


class PartitionTests(unittest.TestCase):
    def test_positive_normalized_weights_and_support(self):
        for sigma in [.0005, .004, .02]:
            for staggered in [False, True]:
                raw, _, _, _ = fixture(sigma, .003, staggered)
                ts, mass, info = nodes(raw, 48, allocation='balanced')
                self.assertEqual(len(ts), 48)
                self.assertTrue(all(.3 <= t <= .7 and w > 0 for t, w in ts))
                self.assertAlmostEqual(sum(w for _, w in ts)+mass, 1, places=12)
                self.assertTrue(0 <= mass <= 1)

    def test_static_fallback_equals_gauss(self):
        raw, _, _, _ = scene('static', device='cpu')
        ts, mass, info = nodes(raw, 32, allocation='balanced')
        x, w = np.polynomial.legendre.leggauss(32)
        np.testing.assert_allclose([t for t, _ in ts], .5+.2*x, atol=1e-15)
        np.testing.assert_allclose([v for _, v in ts], w/2, atol=1e-15)
        self.assertEqual(info['intervals'], 1)
        self.assertAlmostEqual(mass, 0)

    def test_empty_support_is_background(self):
        raw, _, _, _ = fixture(.001, .003, True)
        raw['temporal_means'][:] = 10
        ts, mass, _ = nodes(raw, 32, allocation='balanced')
        self.assertEqual(ts, [])
        self.assertEqual(mass, 1.)

    def test_many_intervals_fallback_does_not_drop_support(self):
        raw, _, _, _ = fixture(.001, .003, True)
        ts, mass, info = nodes(raw, 4, allocation='balanced')
        self.assertEqual(info['intervals'], 1)
        self.assertAlmostEqual(mass, 0)
        self.assertEqual(len(ts), 4)


if __name__ == '__main__':
    unittest.main()
