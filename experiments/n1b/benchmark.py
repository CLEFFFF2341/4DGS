"""Bounded fixed-model experiment. Run: python -m experiments.n1b.benchmark.

All full R calls use the same native CUDA entry point. References are streamed;
timing includes lists/proxies/proposals, CUDA synchronization and full-image R.
No training, checkpoint writes, densification, pruning or parameter updates.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .core import select_ids,build_proxy,render_cuda
from .scenes import scene


def sync(): torch.cuda.synchronize()


def quadrature(render,n,lo,hi,kind='gauss'):
    if kind=='gauss':
        x,w=np.polynomial.legendre.leggauss(n); ts=lo+(x+1)*(hi-lo)/2; weights=w/2
    else:
        if n%2==0: n-=1
        ts=np.linspace(lo,hi,n); weights=np.ones(n); weights[1:-1:2]=4; weights[2:-1:2]=2
        weights/=3*(n-1)
    result=0
    for t,w in zip(ts,weights): result=result+float(w)*render(float(t))
    return result


class Proposal:
    def __init__(self,p):
        self.p=p
        from .core import gaussian_average
        A=gaussian_average(p.a,p.b,p.d,p.duration)*p.mask
        self.total=A.sum()
        self.mass=A.flatten()/self.total
        self.mu=(p.b/p.d).flatten()
        self.sd=p.d.rsqrt().flatten()
        self.lo=torch.special.ndtr((-p.duration/2-self.mu)/self.sd)
        self.hi=torch.special.ndtr((p.duration/2-self.mu)/self.sd)

    def sample(self,k):
        p=self.p
        idx=torch.multinomial(self.mass,k,replacement=True)
        u=torch.rand(k,device=p.a.device,dtype=torch.double)
        prob=self.lo[idx]+u*(self.hi[idx]-self.lo[idx])
        h=self.mu[idx]+self.sd[idx]*torch.special.ndtri(prob.clamp(1e-15,1-1e-15))
        uniform=(torch.rand(k,device=p.a.device,dtype=torch.double)-.5)*p.duration
        h=torch.where(torch.rand(k,device=p.a.device)<.1,uniform,h)
        q=torch.stack([.1/p.duration+.9*p.alpha(t).sum()/(self.total*p.duration) for t in h])
        return h,1/(p.duration*q)


def adaptive_linear(render,lo,hi):
    """Nine-call adaptive piecewise degree-1 CV (not full published Crespo code)."""
    knots=list(np.linspace(lo,hi,5)); vals=[render(t) for t in knots]
    for _ in range(4):
        slopes=[(vals[i+1]-vals[i])/(knots[i+1]-knots[i]) for i in range(len(knots)-1)]
        scores=[]
        for i,s in enumerate(slopes):
            curvature=(s-slopes[max(0,i-1)]).square().mean()+(s-slopes[min(len(slopes)-1,i+1)]).square().mean()
            scores.append(float(curvature)*(knots[i+1]-knots[i])**3)
        i=int(np.argmax(scores)); mid=(knots[i]+knots[i+1])/2
        knots.insert(i+1,mid); vals.insert(i+1,render(mid))
    G=sum((knots[i+1]-knots[i])*(vals[i]+vals[i+1])/2 for i in range(len(knots)-1))/(hi-lo)
    def value(t):
        i=min(max(np.searchsorted(knots,t)-1,0),len(knots)-2)
        f=(t-knots[i])/(knots[i+1]-knots[i])
        return (1-f)*vals[i]+f*vals[i+1]
    return G,value


def evaluate_method(method,raw,cam,pix,bg,render,k):
    lo,hi=.3,.7; duration=hi-lo
    if method in ('gauss','simpson'): return quadrature(render,k,lo,hi,method)
    if method=='adaptive_linear':
        G,g=adaptive_linear(render,lo,hi); k-=9
    elif method=='linear':
        left,right=render(lo),render(hi); k-=2; G=(left+right)/2
        g=lambda t:left+(right-left)*((t-lo)/duration)
    elif method in ('n1b','single_cv','pulse_mc','pulse_cv'):
        ids,_=select_ids(raw,cam,pix,.5,duration)
        p=build_proxy(raw,cam,pix,ids,.5,duration,bg)
        # Importance MC needs only the single-pulse proposal, not a pair integral.
        if method!='pulse_mc':
            G=p.integral(pair=method!='single_cv')
            g=lambda t:p.value(t-.5,pair=method!='single_cv')
    if method in ('pulse_mc','pulse_cv'):
        h,w=Proposal(p).sample(k); times=(h+.5).tolist()
    elif method=='stratified':
        times=(lo+(torch.arange(k,device='cuda')+torch.rand(k,device='cuda'))*duration/k).tolist(); w=[1]*k
    else:
        times=(lo+torch.rand(k,device='cuda')*duration).tolist(); w=[1]*k
    if method in ('uniform','stratified','pulse_mc'):
        return sum(render(t)*ww for t,ww in zip(times,w))/k
    return G+sum((render(t)-g(t))*ww for t,ww in zip(times,w))/k


def gradient_diagnostic(raw,cam,pix,bg,render,p,seeds=48):
    # Fresh graph per sample; G and g differentiated in the same raw coordinates.
    rr={k:v.detach().requires_grad_() for k,v in raw.items()}
    ids,_=select_ids(rr,cam,pix,.5,.4)
    def gradients(value):
        return [g.detach().flatten().double() for g in torch.autograd.grad(value,tuple(rr.values()))]
    rgs=[]; egs=[]
    torch.manual_seed(713)
    sync(); start=time.perf_counter()
    for t in (.3+.4*torch.rand(seeds,device='cuda')).tolist():
        r=render_cuda(rr,cam,t,bg)[:,12:20,12:20].mean()
        rg=gradients(r)
        proxy=build_proxy(rr,cam,pix,ids,.5,.4,bg)
        gg=gradients(proxy.value(t-.5).mean())
        rgs.append(rg); egs.append([a-b for a,b in zip(rg,gg)])
    sync(); elapsed=time.perf_counter()-start
    proxy=build_proxy(rr,cam,pix,ids,.5,.4,bg)
    Gg=gradients(proxy.integral().mean())
    result={}
    for j,key in enumerate(rr):
        r=torch.stack([v[j] for v in rgs]); e=torch.stack([v[j] for v in egs])
        vr=float(r.var(0).sum()); ve=float(e.var(0).sum())
        result[key]={'R_variance_trace':vr,'residual_variance_trace':ve,
                     'ratio':ve/vr if vr>1e-18 else None,
                     'cv_mean_norm':float((Gg[j]+e.mean(0)).norm())}
    return {'samples':seeds,'seconds':elapsed,'groups':result,
            'scope':'branch AD variance, same times; not event-boundary derivatives or training loss gradients'}


def run(args):
    torch.set_num_threads(4); torch.manual_seed(20261019)
    output={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'calls':args.calls,
            'repeats':args.repeats,'baseline':'cc380cd03fd4b290baa74f2acb0a2a2e4470862f','scenes':{},
            'stop_rule':'Stop expansion to reconstruction training if dense/static-layer stress variance ratio > 4; retain diagnostics only.'}
    methods=['uniform','stratified','gauss','simpson','pulse_mc','single_cv','linear','adaptive_linear','n1b','pulse_cv']
    for name in ['single','double','triple','static','static_layers','dense','perspective','overflow']:
        raw,cam,pix,bg=scene(name)
        def render(t): return render_cuda(raw,cam,t,bg)[:,12:20,12:20].permute(1,2,0).reshape(-1,3).double()
        with torch.no_grad():
            render(.5); sync(); torch.cuda.reset_peak_memory_stats()
            ids,stats=select_ids(raw,cam,pix,.5,.4)
            p=build_proxy(raw,cam,pix,ids,.5,.4,bg)
            ref0=quadrature(render,1025,.3,.7,'simpson')
            ref=quadrature(render,2049,.3,.7,'simpson')
            stats['reference_rms_change']=float((ref-ref0).square().mean().sqrt())
            stats['reference_nodes']=2049
            rows={}
            for method in methods:
                values=[]; seconds=[]
                evaluate_method(method,raw,cam,pix,bg,render,args.calls); sync()
                for seed in range(args.repeats):
                    torch.manual_seed(seed+103)
                    sync(); start=time.perf_counter()
                    value=evaluate_method(method,raw,cam,pix,bg,render,args.calls)
                    sync(); seconds.append(time.perf_counter()-start); values.append(value)
                vv=torch.stack(values); mse=(vv-ref).square().mean()
                seconds=float(np.median(seconds))
                rows[method]={'mse':float(mse),'mean_bias_rms':float((vv.mean(0)-ref).square().mean().sqrt()),
                    'variance':float(vv.var(0).mean()),'median_seconds':seconds,'mse_times_seconds':float(mse)*seconds}
            ts=torch.linspace(.3,.7,257,device='cuda')
            rv=torch.stack([render(float(t)) for t in ts]); gv=torch.stack([p.value(t-.5) for t in ts])
            vr=float(rv.var(0).mean()); ve=float((rv-gv).var(0).mean())
            stats.update({'R_variance':vr,'residual_variance':ve,'variance_ratio':ve/vr if vr>1e-18 else None,
                          'proxy_min':float(gv.min()),'proxy_max':float(gv.max()),
                          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            # IID same-wall estimate: floor the measured CV time / per-R time.
            n=max(1,int(args.calls*rows['n1b']['median_seconds']/rows['uniform']['median_seconds']))
            stats['uniform_calls_in_n1b_time']=n
            stats['iid_same_wall_variance_ratio']=(ve/args.calls)/(vr/n) if vr>1e-18 else None
        grad=gradient_diagnostic(raw,cam,pix,bg,render,p)
        output['scenes'][name]={'diagnostics':stats,'methods':rows,'gradients':grad}
        print(name,json.dumps({'variance_ratio':stats['variance_ratio'],'same_wall_ratio':stats['iid_same_wall_variance_ratio'],
                              'N1b_mse':rows['n1b']['mse'],'strat_mse':rows['stratified']['mse']}),flush=True)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(output,indent=2),encoding='utf-8')
    output['decision']='STOP_BEFORE_TRAINING' if any(
        (output['scenes'][s]['diagnostics']['variance_ratio'] or 0)>4 for s in ['static_layers','dense']) else 'REVIEW_BEFORE_EXPANSION'
    Path(args.output).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(output['decision'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--calls',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=24)
    parser.add_argument('--output',default='experiments/n1b/results/benchmark.json')
    args=parser.parse_args()
    if args.calls<10 or args.repeats<2: parser.error('need >=10 calls and >=2 repeats')
    run(args)
