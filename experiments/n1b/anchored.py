"""Piecewise second-order alpha expansion around full midpoint transmittance.

Intervals are fixed independently of residual samples. Each local polynomial in
Gaussian alphas is integrated analytically; no numerical integral substitutes G.
"""
import torch
from dataclasses import replace
from .core import gaussian_average


class AnchoredProxy:
    def __init__(self, proxy, segments=8):
        if not isinstance(segments,int) or segments<1:
            raise ValueError('segments must be a positive integer')
        # Remove globally empty trailing slots; selected IDs/order are unchanged.
        used=max(1,int(proxy.mask.sum(-1).max()))
        proxy=replace(proxy,a=proxy.a[...,:used],b=proxy.b[...,:used],d=proxy.d[...,:used],
                      color=proxy.color[...,:used,:],mask=proxy.mask[...,:used])
        self.proxy=proxy
        self.segments=segments
        self.duration=proxy.duration
        self.width=self.duration/segments
        self.centers=(torch.arange(segments,device=proxy.a.device,dtype=proxy.a.dtype)+.5)*self.width-self.duration/2
        h=self.centers[:,None,None]
        self.anchor=proxy.alpha(h)
        alpha=self.anchor
        one_minus=1-alpha
        trans=torch.cat((torch.ones_like(alpha[...,:1]),one_minus.cumprod(-1)[...,:-1]),-1)
        # Behind-layer color, including background, evaluated by the exact recurrence.
        behind=proxy.background.expand(segments,*proxy.a.shape[:-1],3)
        derivatives=[]; tails=[]
        for i in range(proxy.a.shape[-1]-1,-1,-1):
            tails.append(behind)
            derivatives.append(trans[...,i,None]*(proxy.color[...,i,:]-behind))
            behind=alpha[...,i,None]*proxy.color[...,i,:]+one_minus[...,i,None]*behind
        self.base=behind
        self.first=torch.stack(derivatives[::-1],-2)
        tails=tails[::-1]
        self.front,self.back=torch.triu_indices(proxy.a.shape[-1],proxy.a.shape[-1],offset=1,device=proxy.a.device)
        # Stable product excluding j: no division by (1-alpha_j) at opaque layers.
        pair=[]
        for i in range(1,proxy.a.shape[-1]):
            factors=one_minus[...,:i]
            left=torch.cat((torch.ones_like(factors[...,:1]),factors.cumprod(-1)[...,:-1]),-1)
            rev=factors.flip(-1)
            right=torch.cat((torch.ones_like(rev[...,:1]),rev.cumprod(-1)[...,:-1]),-1).flip(-1)
            tail=tails[i]
            pair.append(-(left*right)[...,None]*(proxy.color[...,i,None,:]-tail[...,None,:]))
        # Above loop is back-major; store indices in the same order.
        self.front=torch.cat([torch.arange(i,device=proxy.a.device) for i in range(1,proxy.a.shape[-1])]) if pair else self.front
        self.back=torch.cat([torch.full((i,),i,device=proxy.a.device,dtype=torch.long) for i in range(1,proxy.a.shape[-1])]) if pair else self.back
        self.second=torch.cat(pair,-2) if pair else self.first[...,:0,:]

    def value(self,h,pair=True):
        index=int((float(h)+self.duration/2)/self.width)
        index=max(0,min(index,self.segments-1))
        delta=self.proxy.alpha(h)-self.anchor[index]
        value=self.base[index]+(self.first[index]*delta[...,None]).sum(-2)
        if pair:
            value=value+(self.second[index]*(delta[...,self.front]*delta[...,self.back])[...,None]).sum(-2)
        return value

    def integral(self,pair=True):
        p=self.proxy; h=self.centers[:,None,None]
        a=p.a+p.b*h-.5*p.d*h*h
        b=p.b-p.d*h
        A=gaussian_average(a,b,p.d,self.width)*p.mask
        value=self.base+(self.first*(A-self.anchor)[...,None]).sum(-2)
        if pair and len(self.front):
            j,i=self.front,self.back
            Aij=gaussian_average(a[...,j]+a[...,i],b[...,j]+b[...,i],p.d[...,j]+p.d[...,i],self.width)
            Aij=Aij*p.mask[...,j]*p.mask[...,i]
            cross=Aij-A[...,j]*self.anchor[...,i]-A[...,i]*self.anchor[...,j]+self.anchor[...,j]*self.anchor[...,i]
            value=value+(self.second*cross[...,None]).sum(-2)
        return value.mean(0)
