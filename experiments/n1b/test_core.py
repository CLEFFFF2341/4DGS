import unittest
import torch
from .core import gaussian_average,select_ids,build_proxy,render_cuda
from .scenes import scene


class CoreTests(unittest.TestCase):
    def test_integral_tails_and_flat_limit(self):
        import scipy.integrate
        import math
        for mu,sigma in [(0,1),(.15,.005),(1,.1),(-1,.1),(0,1e7)]:
            a=torch.tensor(-.5*(mu/sigma)**2,dtype=torch.double)
            b=torch.tensor(mu/sigma**2,dtype=torch.double)
            d=torch.tensor(1/sigma**2,dtype=torch.double)
            expected=scipy.integrate.quad(lambda h: math.exp(-.5*((h-mu)/sigma)**2),-.5,.5,
                epsabs=1e-14,points=[mu] if abs(mu)<.5 else None)[0]
            self.assertAlmostEqual(float(gaussian_average(a,b,d,1)),expected,delta=max(1e-14,expected*1e-9))
        z=torch.tensor(0.,dtype=torch.double,requires_grad=True)
        out=gaussian_average(z,z,z,1)
        self.assertTrue(torch.isfinite(torch.autograd.grad(out,z)[0]))

    def test_pair_exact_for_two_local_layers(self):
        raw,cam,pix,bg=scene('double','cpu')
        ids,_=select_ids(raw,cam,pix,.5,.4)
        p=build_proxy(raw,cam,pix,ids,.5,.4,bg)
        for h in [-.2,0,.17]:
            torch.testing.assert_close(p.value(h),p.full_local(h),atol=1e-12,rtol=1e-12)
        h=torch.linspace(-.2,.2,2001,dtype=torch.double)
        numeric=torch.stack([p.value(t) for t in h])
        torch.testing.assert_close(p.integral(),torch.trapezoid(numeric,h,dim=0)/.4,atol=2e-7,rtol=2e-6)

    def test_chunk_independence_and_cap(self):
        raw,cam,pix,bg=scene('overflow','cpu')
        one,stats=select_ids(raw,cam,pix,.5,.4,primitive_chunk=3,pixel_chunk=7)
        two,_=select_ids(raw,cam,pix,.5,.4,primitive_chunk=100,pixel_chunk=100)
        self.assertTrue(torch.equal(one,two)); self.assertGreater(stats['overflow_pixels'],0)
        raw['spatial_means'][:,2]=-1
        ids,stats=select_ids(raw,cam,pix,.5,.4)
        self.assertTrue((ids==-1).all())
        p=build_proxy(raw,cam,pix,ids,.5,.4,bg)
        torch.testing.assert_close(p.integral(),bg.double().expand(len(pix),3))
        raw['spatial_means'][:,2]=3
        raw['left_isoclinic_rotations'].zero_()
        ids,stats=select_ids(raw,cam,pix,.5,.4)
        self.assertTrue((ids==-1).all())
        p=build_proxy(raw,cam,pix,ids,.5,.4,bg)
        torch.testing.assert_close(p.integral(),bg.double().expand(len(pix),3))

    def test_dense_pair_integral_and_unclipped_estimator(self):
        raw,cam,pix,bg=scene('dense','cpu')
        ids,_=select_ids(raw,cam,pix[:1],.5,.4)
        p=build_proxy(raw,cam,pix[:1],ids,.5,.4,bg)
        h=torch.linspace(-.2,.2,4001,dtype=torch.double)
        values=torch.stack([p.value(t) for t in h])
        self.assertLess(float(values.min()),0)
        torch.testing.assert_close(p.integral(),torch.trapezoid(values,h,dim=0)/.4,atol=3e-6,rtol=1e-6)
        # A negative degree-two proxy is not itself a bug: its integral cancels
        # the average residual even under severe multi-layer occlusion.
        full=torch.stack([p.full_local(t) for t in h])
        corrected=p.integral()+torch.trapezoid(full-values,h,dim=0)/.4
        torch.testing.assert_close(corrected,torch.trapezoid(full,h,dim=0)/.4,atol=3e-6,rtol=1e-6)

    def test_raw65_directional_derivatives(self):
        raw,cam,pix,bg=scene('double','cpu')
        torch.manual_seed(17)
        raw={k:v.double().requires_grad_() for k,v in raw.items()}
        raw['sh_coefficients_rest'].data.normal_(0,.03)
        ids,_=select_ids(raw,cam,pix[:2],.5,.4)
        def func(rr): return build_proxy(rr,cam,pix[:2],ids,.5,.4,bg).integral().sum()
        value=func(raw); grads=torch.autograd.grad(value,tuple(raw.values()))
        for (key,param),grad in zip(raw.items(),grads):
            direction=torch.randn_like(param); direction/=direction.norm()
            eps=1e-5
            plus=dict(raw); minus=dict(raw)
            plus[key]=param+eps*direction; minus[key]=param-eps*direction
            fd=(func(plus)-func(minus))/(2*eps)
            torch.testing.assert_close((grad*direction).sum(),fd,atol=2e-7,rtol=2e-4,msg=key)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_matches_static_proxy_and_raw_gradients(self):
        raw,cam,pix,bg=scene('single')
        raw['left_isoclinic_rotations'][:]=torch.tensor([1,0,0,0],device='cuda')
        raw['temporal_scales'][:]=14.
        raw['sh_coefficients_rest'].normal_(0,.02)
        raw={k:v.requires_grad_() for k,v in raw.items()}
        ids,_=select_ids(raw,cam,pix,.5,.4)
        p=build_proxy(raw,cam,pix,ids,.5,.4,bg)
        image=render_cuda(raw,cam,.5,bg)[:,12:20,12:20].permute(1,2,0).reshape(-1,3)
        torch.testing.assert_close(image.double(),p.value(0),atol=3e-6,rtol=2e-5)
        grads=torch.autograd.grad(image.sum(),tuple(raw.values()))
        pg=torch.autograd.grad(p.value(0).sum(),tuple(raw.values()))
        for key,g,h in zip(raw,grads,pg):
            torch.testing.assert_close(g,h,atol=4e-4,rtol=2e-4,msg=key)


if __name__=='__main__': unittest.main()
