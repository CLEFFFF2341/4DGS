"""Representation equivalence and derivative invariants for new structures."""
import unittest
import torch
from .models import TemporalModel, velocity, predict_state
from experiments.n1b_v2.real_dynamic import DynamicData, predict


@unittest.skipUnless(torch.cuda.is_available(), 'native GPU runtime required')
class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=DynamicData('dataset/dsec_dynamic/zurich_city_04_c',120)
        cls.raw={k:v[:128].clone() for k,v in torch.load('output/dsec_c120_s0/shared_initial.pt',weights_only=True).items()}

    def test_zero_initialization_and_saved_render(self):
        expected,_=predict(self.raw,self.data,0,0,2,'gauss')
        for kind in ('native','anchor','warp','curve','affine_curve','affine_spline','translation_spline','affine_linear'):
            m=TemporalModel(self.raw,kind)
            actual=m.predict(self.data,0,0,2)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(predict_state(m.state(),self.data,0,0,2),actual,atol=0,rtol=0)

    def test_anchor_conditional_path_invariance(self):
        m=TemporalModel(self.raw,'anchor');before=m.base_raw();v=velocity(before)
        t=.6;old=before['spatial_means']+v*(t-before['temporal_means'])
        with torch.no_grad():m.joint.fixed['temporal_means'].add_(.02)
        after=m.base_raw();new=after['spatial_means']+velocity(after)*(t-after['temporal_means'])
        torch.testing.assert_close(old,new,atol=1e-5,rtol=2e-6)

    def test_warp_monotonicity_and_endpoints(self):
        m=TemporalModel(self.raw,'warp')
        for a,b in ((10.,10.),(-10.,10.),(10.,-10.),(-10.,-10.)):
            with torch.no_grad():m.extra.copy_(torch.tensor([a,b],device='cuda'))
            t=torch.linspace(0,1,101,device='cuda',requires_grad=True)
            aa,bb=.45*m.extra.tanh();w=t+aa*t*(1-t)+bb*t*(1-t)*(2*t-1)
            derivative=torch.autograd.grad(w.sum(),t)[0]
            self.assertGreater(float(derivative.min()),.09)
            self.assertEqual(float(w[0]),0.);self.assertEqual(float(w[-1]),1.)

    def test_extra_gradient_finite_and_nonzero(self):
        for kind in ('warp','curve','affine_curve','affine_spline','translation_spline','affine_linear'):
            m=TemporalModel(self.raw,kind)
            loss=m.predict(self.data,0,0,1).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(m.extra.grad).all())
            self.assertGreater(float(m.extra.grad.abs().max()),0.)


if __name__=='__main__':unittest.main()
