"""Validate original loss wiring and single-time baseline semantics."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.n1b_v2.real_fastergs import load_original_loss,midpoint,original_defaults,update_spatial_lr


class FasterGSCoreTests(unittest.TestCase):
    def test_exact_single_timestamp(self):
        data=SimpleNamespace(bounds=np.array([[[.12,.28]]]),cameras=['camera'],bg='background')
        with patch('experiments.n1b_v2.real_fastergs.render_cuda',return_value='image') as render:
            self.assertEqual(midpoint('raw',data,0,0),'image')
            render.assert_called_once_with('raw','camera',.2,'background')

    @unittest.skipUnless(torch.cuda.is_available(),'native fused SSIM required')
    def test_unmodified_original_loss_and_gradient(self):
        from fused_ssim import fused_ssim
        original,defaults=load_original_loss(Path('output'))
        torch.manual_seed(8);a=torch.rand(3,32,32,device='cuda',requires_grad=True);b=torch.rand_like(a)
        actual=original(a,b)
        expected=.8*(a-b).abs().mean()+.2*(1-fused_ssim(a[None],b[None]))
        torch.testing.assert_close(actual,expected)
        g1=torch.autograd.grad(actual,a,retain_graph=True)[0];g2=torch.autograd.grad(expected,a)[0]
        torch.testing.assert_close(g1,g2)

    def test_original_schedule_updates_spatial_means_only(self):
        d=original_defaults();extent=.3
        opt=SimpleNamespace(param_groups=[{'name':'spatial_means','lr':.1},{'name':'temporal_means','lr':.123}])
        update_spatial_lr(opt,d,extent,30000)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],.0000016*extent)
        self.assertEqual(opt.param_groups[1]['lr'],.123)


if __name__=='__main__':unittest.main()
