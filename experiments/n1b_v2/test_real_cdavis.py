"""Check the scalable adapter against the archived node policy."""
import unittest
import numpy as np
import torch
from experiments.n1b_v2.partition import nodes
from experiments.n1b_v2.real_cdavis import schedule


@unittest.skipUnless(torch.cuda.is_available(), 'native CUDA environment required')
class ScheduleTest(unittest.TestCase):
    def test_exact_archived_policy(self):
        for seed in range(40):
            torch.manual_seed(seed)
            raw = {'spatial_scales':torch.randn(25,3,device='cuda')-4,
                   'temporal_scales':torch.randn(25,1,device='cuda')-3,
                   'temporal_means':torch.rand(25,1,device='cuda'),
                   'left_isoclinic_rotations':torch.randn(25,4,device='cuda'),
                   'right_isoclinic_rotations':torch.randn(25,4,device='cuda')}
            for calls in (8,32):
                a, am, _ = schedule(raw,.3,.7,calls,'partition')
                b, bm, _ = nodes(raw,calls,.3,.7,allocation='balanced')
                np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
                self.assertAlmostEqual(am,bm,places=12)


if __name__ == '__main__': unittest.main()
