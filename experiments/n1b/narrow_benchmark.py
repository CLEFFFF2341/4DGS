"""Narrow-pulse falsification test; synchronized timed forward and raw VJP estimates.

Counts chosen by calibration only, fixed before accuracy seeds. Over-budget trials
are explicitly ineligible, never silently relabelled equal-time comparisons.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .scenes import scene
from .core import build_proxy,select_ids,render_cuda
from .anchored import AnchoredProxy
from .benchmark import Proposal,sync


def fixture(sigma,phase,staggered):
    raw,cam,pix,bg=scene('single')
    raw={k:v.repeat(8,*([1]*(v.ndim-1))) for k,v in raw.items()}
    raw['left_isoclinic_rotations'][:]=raw['left_isoclinic_rotations'].new_tensor([1,0,0,0])
    raw['spatial_means'][:,0]=(torch.arange(8,device='cuda')+12.5-cam.cx)*3/cam.fx
    raw['spatial_means'][:,1]=0
    raw['spatial_scales'][:,0]=math.log(.025)
    raw['temporal_scales'][:]=math.log(sigma)
    peaks=torch.linspace(.34,.66,8,device='cuda') if staggered else torch.full((8,),.5,device='cuda')
    raw['temporal_means'][:,0]=peaks+phase
    return raw,cam,pix,bg


def proxy(raw,cam,pix,bg):
    ids,_=select_ids(raw,cam,pix,.5,.4)
    return build_proxy(raw,cam,pix,ids,.5,.4,bg)


def event_nodes(raw,cam,pix,bg,order):
    # All alpha/time cull boundaries for the ROI, for this stationary geometry.
    # Nodes are detached: reference is the expectation of backend branch AD.
    with torch.no_grad():
        p=proxy(raw,cam,pix,bg)
        mu=p.b/p.d
        peak=p.a+p.b.square()/(2*p.d)
        radius=(2*(peak-math.log(1/255)).clamp_min(0)/p.d).sqrt()
        cuts=[.3,.7]
        for t in (mu-radius+.5,mu+radius+.5): cuts+=t[p.mask].flatten().tolist()
        tau=raw['temporal_means'][:,0]; sigma=raw['temporal_scales'][:,0].exp()
        cuts+=(tau-math.sqrt(-2*math.log(.05))*sigma).tolist()
        cuts+=(tau+math.sqrt(-2*math.log(.05))*sigma).tolist()
        cuts=sorted(set(float(t) for t in cuts if .3<=t<=.7))
    x,w=np.polynomial.legendre.leggauss(order)
    return [(float((l+r)/2+xx*(r-l)/2),float(ww*(r-l)/.8))
            for l,r in zip(cuts[:-1],cuts[1:]) for xx,ww in zip(x,w)]


def render(raw,cam,bg,t):
    return render_cuda(raw,cam,t,bg)[:,12:20,12:20].permute(1,2,0).reshape(-1,3).double()


def vjp(value,raw,weight,retain=False):
    grads=torch.autograd.grad((value*weight).sum(),tuple(raw.values()),retain_graph=retain)
    return torch.cat([g.flatten().double() for g in grads])


def reference(raw,cam,pix,bg,weight,order):
    nodes=event_nodes(raw,cam,pix,bg,order)
    image=0; grad=0
    for t,w in nodes:
        r=render(raw,cam,bg,t)
        image=image+w*r.detach(); grad=grad+w*vjp(r,raw,weight)
    return image,grad,len(nodes)


def estimate(method,k,raw,cam,pix,bg,weight,gradient):
    result=0; p=None
    if method in ('anchored','single','pulse'):
        p=proxy(raw,cam,pix,bg)
        if method=='anchored': p=AnchoredProxy(p)
        if method!='pulse':
            G=p.integral(pair=method!='single')
            result=vjp(G,raw,weight,True) if gradient else G
    if method=='gauss':
        x,w=np.polynomial.legendre.leggauss(k)
        nodes=list(zip(.5+.2*x,w/2))
    elif method=='pulse':
        with torch.no_grad(): h,w=Proposal(p).sample(k)
        nodes=list(zip((h+.5).tolist(),(w/k).tolist()))
    else:
        # Same stratified design for baseline and CV; no sample reuse in fitting.
        times=.3+.4*(torch.arange(k,device='cuda')+torch.rand(k,device='cuda'))/k
        nodes=[(t,1/k) for t in times.tolist()]
    for t,w in nodes:
        r=render(raw,cam,bg,t)
        if method in ('anchored','single'): r=r-p.value(t-.5,pair=method!='single')
        result=result+w*(vjp(r,raw,weight,True) if gradient else r)
    return result.detach()


def timed(*args):
    sync(); start=time.perf_counter()
    with torch.set_grad_enabled(args[-1]): value=estimate(*args)
    sync(); return value,time.perf_counter()-start


def calibrate(method,budget,raw,cam,pix,bg,weight,gradient):
    # Include setup/backward costs. Three timed checks; leave 20% headroom.
    k=8
    for _ in range(3):
        _,elapsed=timed(method,k,raw,cam,pix,bg,weight,gradient)
        k=max(1,min(256,int(k*budget*.8/max(elapsed,1e-6))))
    return k


def run(args):
    torch.set_num_threads(4)
    out={'device':torch.cuda.get_device_name(0),'repeats':args.repeats,'budget_seconds':args.budget,
         'gradient_target':'fixed random spatial RGB VJP, raw65 per primitive; expected backend branch AD, not boundary derivatives',
         'protocol':'calibration fixes counts before seeds; same-call 16 plus actual timed budget; every elapsed and error retained; over-budget trials ineligible',
         'cases':{}}
    methods=['stratified','gauss','pulse','single','anchored']
    for sigma in [.016,.004,.001]:
      for staggered in [False,True]:
       for phase in [-.003,.003]:
        name=f'{sigma}_{staggered}_{phase}'
        raw,cam,pix,bg=fixture(sigma,phase,staggered)
        raw={k:v.requires_grad_() for k,v in raw.items()}
        gen=torch.Generator(device='cuda').manual_seed(671)
        weight=torch.randn((64,3),generator=gen,device='cuda',dtype=torch.double)/192
        r0,g0,n0=reference(raw,cam,pix,bg,weight,4)
        ref,gr,n=reference(raw,cam,pix,bg,weight,8)
        with torch.no_grad():
            p=proxy(raw,cam,pix,bg)
            # Peak of total alpha identifies actual pixel-wise pulse offset.
            grid=torch.linspace(.3,.7,2049,device='cuda',dtype=torch.double)
            curves=torch.stack([p.alpha(t-.5).sum(-1) for t in grid])
            peak_times=grid[curves.argmax(0)]
            proposal=Proposal(p)
            q=torch.stack([.1/.4+.9*p.alpha(t-.5).sum()/(proposal.total*.4) for t in grid])
        case={'sigma':sigma,'phase':phase,'staggered':staggered,'reference_nodes':[n0,n],
              'reference_image_mse_change':float((ref-r0).square().mean()),
              'reference_gradient_mse_change':float((gr-g0).square().mean()),
              'reference_gradient_norm':float(gr.norm()),'pixel_peak_times':peak_times.tolist(),
              'proposal_min_max':[float(q.min()),float(q.max())], 'measurements':{}}
        for gradient in [False,True]:
          for regime in ['same_calls','same_time']:
            rows={}
            for method in methods:
                k=16 if regime=='same_calls' else calibrate(method,args.budget,raw,cam,pix,bg,weight,gradient)
                timed(method,k,raw,cam,pix,bg,weight,gradient)
                trials=[]
                for seed in range(args.repeats):
                    torch.manual_seed(seed+173)
                    value,elapsed=timed(method,k,raw,cam,pix,bg,weight,gradient)
                    target=gr if gradient else ref
                    trials.append({'seconds':elapsed,'mse':float((value-target).square().mean()),'within_budget':elapsed<=args.budget})
                rows[method]={'calls':k,'mean_mse':float(np.mean([t['mse'] for t in trials])),
                    'median_seconds':float(np.median([t['seconds'] for t in trials])),
                    'all_within_budget':all(t['within_budget'] for t in trials),'trials':trials}
            case['measurements'][f'{"gradient" if gradient else "image"}_{regime}']=rows
        out['cases'][name]=case
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
        print(name,json.dumps({kind:{m:round(v['mean_mse'],12) for m,v in rows.items()} for kind,rows in case['measurements'].items() if kind.endswith('same_time')}),flush=True)
    # Must beat every eligible simple comparator in BOTH quantities, both phases.
    wins=[]
    for name,c in out['cases'].items():
        passed=True
        for mode in ['image_same_time','gradient_same_time']:
            rows=c['measurements'][mode]; cv=rows['anchored']
            others=[rows[m] for m in ['stratified','gauss','pulse','single'] if rows[m]['all_within_budget']]
            passed &= cv['all_within_budget'] and len(others)==4 and cv['mean_mse']<.8*min(r['mean_mse'] for r in others)
        if passed: wins.append(name)
    out['strict_joint_wins']=wins
    out['decision']='STOP_NO_ROBUST_ADVANTAGE' if not any(
        all(f'{sigma}_{staggered}_{phase}' in wins for phase in [-.003,.003])
        for sigma in [.016,.004,.001] for staggered in [False,True]) else 'BENEFIT_FOUND_REVIEW_BEFORE_STEP_2_3'
    Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(out['decision'],wins,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=12)
    parser.add_argument('--budget',type=float,default=.05)
    parser.add_argument('--output',default='experiments/n1b/results/narrow.json')
    args=parser.parse_args()
    if args.repeats<2 or args.budget<=0:
        parser.error('need at least two repeats and a positive time budget')
    run(args)
