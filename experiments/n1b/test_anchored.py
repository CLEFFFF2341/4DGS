import unittest
import torch
from .core import select_ids,build_proxy
from .anchored import AnchoredProxy
from .scenes import scene


class AnchoredTests(unittest.TestCase):
    def make(self,name,pixels=2,segments=8):
        raw,cam,pix,bg=scene(name,'cpu')
        ids,_=select_ids(raw,cam,pix[:pixels],.5,.4)
        p=build_proxy(raw,cam,pix[:pixels],ids,.5,.4,bg)
        return p,AnchoredProxy(p,segments)

    def test_two_layers_exact_and_static_foreground_preserved(self):
        for name in ['double','static_layers']:
            p,q=self.make(name)
            for h in [-.2,-.123,0,.023,.2]:
                torch.testing.assert_close(q.value(h),p.full_local(h),atol=2e-12,rtol=2e-12)

    def test_dense_integral_matches_piecewise_quadrature(self):
        import numpy as np
        p,q=self.make('dense')
        x,w=np.polynomial.legendre.leggauss(32)
        total=0
        for c in q.centers:
            for t,weight in zip(x,w): total=total+q.value(float(c)+t*q.width/2)*weight/(2*q.segments)
        torch.testing.assert_close(q.integral(),total,atol=2e-12,rtol=2e-12)

    def test_fully_opaque_anchor_has_finite_gradients(self):
        from dataclasses import replace
        p,_=self.make('triple')
        p=replace(p,a=torch.zeros_like(p.a,requires_grad=True),b=torch.zeros_like(p.b),d=torch.zeros_like(p.d))
        q=AnchoredProxy(p)
        result=q.integral()
        torch.testing.assert_close(result,p.color[:,0,:],atol=2e-12,rtol=2e-12)
        self.assertTrue(torch.isfinite(torch.autograd.grad(result.sum(),p.a)[0]).all())

    def test_raw_gradient_finite_difference(self):
        for name in ['static_layers','dense']:
            with self.subTest(scene=name): self.check_raw_gradient(name)

    def check_raw_gradient(self,name):
        raw,cam,pix,bg=scene(name,'cpu')
        torch.manual_seed(37)
        raw={k:v.double().requires_grad_() for k,v in raw.items()}
        ids,_=select_ids(raw,cam,pix[:1],.5,.4)
        def fn(rr): return AnchoredProxy(build_proxy(rr,cam,pix[:1],ids,.5,.4,bg)).integral().sum()
        grads=torch.autograd.grad(fn(raw),tuple(raw.values()))
        for (key,param),grad in zip(raw.items(),grads):
            direction=torch.randn_like(param); direction/=direction.norm()
            plus=dict(raw); minus=dict(raw); eps=1e-5
            plus[key]=param+eps*direction; minus[key]=param-eps*direction
            fd=(fn(plus)-fn(minus))/(2*eps)
            torch.testing.assert_close((grad*direction).sum(),fd,atol=2e-7,rtol=3e-4,msg=key)


if __name__=='__main__': unittest.main()
