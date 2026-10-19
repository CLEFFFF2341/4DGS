"""Validate the new experiment, including its peak staggering and raw VJP target."""
import unittest
import torch
from .narrow_benchmark import fixture,proxy,event_nodes,estimate


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class NarrowTests(unittest.TestCase):
    def test_peaks_and_reference_weights(self):
        for staggered in [False,True]:
            raw,cam,pix,bg=fixture(.001,.003,staggered)
            p=proxy(raw,cam,pix,bg)
            # At each column center the strongest primitive is its own column.
            strongest=p.a.add(p.b.square()/(2*p.d)).masked_fill(~p.mask,-torch.inf).argmax(-1)
            peaks=(.5+p.b/p.d).gather(-1,strongest[:,None])[:,0]
            span=float(peaks.max()-peaks.min())
            self.assertGreater(span,.3) if staggered else self.assertLess(span,1e-6)
            nodes=event_nodes(raw,cam,pix,bg,4)
            self.assertAlmostEqual(sum(w for _,w in nodes),1.,places=12)
            self.assertTrue(all(.3<=t<=.7 and w>0 for t,w in nodes))

    def test_timed_estimators_return_full_raw_gradient(self):
        raw,cam,pix,bg=fixture(.001,-.003,True)
        raw={k:v.requires_grad_() for k,v in raw.items()}
        weight=torch.ones((64,3),device='cuda',dtype=torch.double)/192
        expected=sum(v.numel() for v in raw.values())
        for method in ['stratified','gauss','pulse','single','anchored']:
            grad=estimate(method,2,raw,cam,pix,bg,weight,True)
            self.assertEqual(grad.numel(),expected)
            self.assertTrue(torch.isfinite(grad).all())


if __name__=='__main__': unittest.main()
