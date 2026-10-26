"""Real measured-shutter CDAVIS validation; no event/deblurred/test image supervision.

The common camera-motion adapter uses the recorded encoder trajectory. Test
images have sentinel timestamps, so the static scene is queried at the time of
the nearest recorded camera position (with interpolation along the slider).
This is a bounded refinement experiment, not the official EvDeblurNeRF protocol.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from scipy.interpolate import interp1d
from torchmetrics.functional.image import structural_similarity_index_measure

from experiments.n1b.core import Camera, render_cuda, rotation
from experiments.n1b_v2.exposure import gauss_rule
from experiments.n1b_v2.joint_optimization import JointParameters


class Dataset:
    def __init__(self, root, scene):
        self.path = Path(root) / scene
        p = np.load(self.path / 'poses_bounds.npy')[:, :15].reshape(-1, 3, 5)
        ap = np.load(self.path / 'all_poses_bounds.npy')[:, :15].reshape(-1, 3, 5)
        tm = np.load(self.path / 'images_1/timestamps.npz')
        self.origin = float(tm['start'])
        self.start = (tm['timestamps_start'].astype(np.float64) - self.origin) / 1e6
        self.end = (tm['timestamps_end'].astype(np.float64) - self.origin) / 1e6
        self.train = np.flatnonzero(self.end > self.start).tolist()
        self.test = list(range(len(p)-5, len(p)))
        assert self.train == list(range(len(p)-5)), 'Require official last-five split'
        self.span = max(self.end[self.train])
        self.times = (np.load(self.path / 'all_timestamps.npy') - self.origin) / 1e6
        self.h, self.w, self.f = p[0, :, 4]
        self.h, self.w = int(self.h), int(self.w)
        # LLFF [down,right,back] -> native [right,down,forward].
        self.rot = np.stack((ap[:, :, 1], ap[:, :, 0], -ap[:, :, 2]), -1)
        assert np.allclose(self.rot, np.eye(3), atol=1e-5), 'This adapter is for the recorded linear slider only'
        self.positions = ap[:, :, 3]
        self.trajectory = interp1d(self.times, self.positions, axis=0, bounds_error=False, fill_value='extrapolate')
        self.cameras = [self.camera_at_position(v[:, 3]) for v in p]
        order = np.argsort(self.positions[:, 0])
        self.test_times = {i:float(np.interp(p[i,0,3],self.positions[order,0],self.times[order])) for i in self.test}
        self.images = [torch.as_tensor(np.asarray(Image.open(self.path/'images_1'/f'{i:02d}.png')).copy(), device='cuda').permute(2,0,1).float()/255 for i in range(len(p))]
        self.bg = torch.zeros(3,device='cuda')

    def camera_at_position(self, pos):
        pos = torch.as_tensor(np.array(pos), device='cuda', dtype=torch.float32)
        w2c = torch.eye(4,device='cuda'); w2c[:3,3] = -pos
        return Camera(w2c,pos,self.w,self.h,float(self.f),float(self.f),self.w/2,self.h/2,.01,100.)

    def camera(self,t):
        return self.camera_at_position(self.trajectory(t))

    def manifest(self):
        return {'scene':self.path.name,'train':self.train,'test':self.test,'resolution':[self.w,self.h],
                'exposure_seconds':(self.end[self.train]-self.start[self.train]).tolist(),
                'trajectory_seconds':[float(self.times.min()),float(self.times.max())],
                'extrapolated_edge_seconds':[max(0.,float(self.times.min()-min(self.start[self.train]))),max(0.,float(max(self.end[self.train])-self.times.max()))],
                'test_times_from_slider_position':self.test_times,
                'file_sha256':{str(p.relative_to(self.path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.path.rglob('*') if p.is_file() and p.suffix in ('.npy','.npz','.png')}}


@torch.no_grad()
def schedule(raw, lo, hi, calls, method):
    x,w = gauss_rule(calls)
    global_rule = list(zip((lo+hi)/2+x*(hi-lo)/2,w/2))
    if method == 'gauss': return global_rule,0.,True
    # Exact interval discovery equivalent to partition.nodes, but O(N log N).
    R=rotation(raw['left_isoclinic_rotations'],raw['right_isoclinic_rotations'])
    variance=(2*torch.cat((raw['spatial_scales'],raw['temporal_scales']),-1)).exp()
    sigma=(R[:,3].square()*variance).sum(-1).sqrt().cpu().numpy()
    center=raw['temporal_means'][:,0].cpu().numpy()
    radius=math.sqrt(-2*math.log(.05))*sigma
    a=(torch.as_tensor(center)-torch.as_tensor(radius)).numpy().astype(float)
    b=(torch.as_tensor(center)+torch.as_tensor(radius)).numpy().astype(float)
    valid=(a<hi)&(b>lo); a=np.maximum(a[valid],lo);b=np.minimum(b[valid],hi)
    if not len(a): return [],1.,False
    cuts=np.unique(np.concatenate(([lo,hi],a,b))); mids=(cuts[1:]+cuts[:-1])/2
    active=np.searchsorted(np.sort(a),mids,side='left')-np.searchsorted(np.sort(b),mids,side='right')>0
    intervals=list(zip(cuts[:-1][active],cuts[1:][active]))
    if len(intervals)>calls//2 or intervals==[(lo,hi)]: return global_rule,0.,True
    counts=[1]*len(intervals)
    scores=[(d-c)/max(float(sigma[np.abs((c+d)/2-center)<=math.sqrt(-2*math.log(.05))*sigma].min()),1e-15) for c,d in intervals]
    for _ in range(calls-len(intervals)):
        j=max(range(len(intervals)),key=lambda j:scores[j]/counts[j]);counts[j]+=1
    out=[]
    for (a,b),n in zip(intervals,counts):
        x,w=gauss_rule(n);out.extend(zip((a+b)/2+x*(b-a)/2,w*(b-a)/(2*(hi-lo))))
    return out,max(0.,1-sum(b-a for a,b in intervals)/(hi-lo)),False


def integrated(raw, data, i, calls, method):
    nodes,mass,equal=schedule(raw,float(data.start[i]),float(data.end[i]),calls,method)
    pred=data.bg[:,None,None].expand(3,data.h,data.w)*mass
    for t,w in nodes: pred=pred+float(w)*render_cuda(raw,data.camera(float(t)),float(t),data.bg)
    return pred,equal


def initialize(data, seed, n):
    g=torch.Generator(device='cuda').manual_seed(seed)
    idx=torch.randint(len(data.train),(n,),generator=g,device='cuda')
    u=torch.rand(n,generator=g,device='cuda')*data.w;v=torch.rand(n,generator=g,device='cuda')*data.h
    z=torch.exp(torch.rand(n,generator=g,device='cuda')*math.log(4/.4)+math.log(.4))
    origins=torch.stack([data.cameras[i].position for i in data.train])[idx]
    means=origins+torch.stack(((u-data.w/2)*z/data.f,(v-data.h/2)*z/data.f,z),-1)
    rgb=torch.stack([data.images[i] for i in data.train])[idx,:,v.long(),u.long()]
    identity=torch.zeros(n,4,device='cuda');identity[:,0]=1
    return {'spatial_means':means,'temporal_means':torch.full((n,1),data.span/2,device='cuda'),
            'spatial_scales':(z[:,None].expand(n,3)/data.f*2).log().contiguous(),
            'temporal_scales':torch.full((n,1),math.log(data.span*2),device='cuda'),
            'left_isoclinic_rotations':identity.clone(),'right_isoclinic_rotations':identity.clone(),
            'opacities':torch.full((n,1),math.log(.1/.9),device='cuda'),
            'sh_coefficients_0':((rgb-.5)/.28209479177387814)[:,None,:].contiguous(),
            'sh_coefficients_rest':torch.zeros(n,15,3,device='cuda')}


def train(data, initial, mode, method, steps, calls, seed, warmup=False):
    joint=JointParameters(initial,mode=mode)
    opt=joint.optimizer(lr=.01 if mode=='legacy' else .05,spatial_lr=.01)
    # Same appearance/shape optimization in every arm. Joint XYZ and time-width
    # parameterization is the experimental factor. No densification or pruning.
    for key,lr in [('sh_coefficients_0',.01),('sh_coefficients_rest',.0005),('opacities',.01),('spatial_scales',.005)]:
        joint.fixed[key].requires_grad_();opt.add_param_group({'params':[joint.fixed[key]],'lr':lr})
    if warmup: joint.temporal.requires_grad_(False)
    rng=np.random.default_rng(seed); order=[]
    while len(order)<steps: order.extend(rng.permutation(data.train).tolist())
    equal_count=0;losses=[];torch.cuda.synchronize();start=time.perf_counter()
    for step,i in enumerate(order[:steps]):
        opt.zero_grad(set_to_none=True)
        pred,equal=integrated(joint.raw(),data,i,calls,method);equal_count+=int(equal)
        loss=(pred-data.images[i]).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward();opt.step();losses.append(float(loss.detach()))
        if (step+1)%100==0:
            print(json.dumps({'stage':'warmup' if warmup else mode+'-'+method,'step':step+1,'loss':losses[-1],'seconds':time.perf_counter()-start}),flush=True)
    torch.cuda.synchronize()
    return {k:v.detach().clone() for k,v in joint.raw().items()}, {'seconds':time.perf_counter()-start,'steps':steps,'global_gauss_identical_schedule_steps':equal_count,'loss_first':losses[0],'loss_last':losses[-1]}


@torch.no_grad()
def evaluate(data, raw, folder, calls):
    folder.mkdir(parents=True,exist_ok=True);rows=[]
    for i in data.test:
        pred=render_cuda(raw,data.cameras[i],data.test_times[i],data.bg).clamp(0,1)
        ref=data.images[i];mse=float((pred-ref).square().mean())
        a=pred.permute(1,2,0).cpu().numpy();b=ref.permute(1,2,0).cpu().numpy()
        rows.append({'index':i,'psnr':-10*math.log10(mse),'ssim':float(structural_similarity_index_measure(pred[None],ref[None],data_range=1.))})
        Image.fromarray((a*255).round().astype(np.uint8)).save(folder/f'{i:02d}.png')
    losses=[];quad=[]
    for i in data.train:
        a,_=integrated(raw,data,i,calls,'gauss');b,_=integrated(raw,data,i,calls*2,'gauss')
        losses.append(float((b-data.images[i]).square().mean()));quad.append(float((a-b).square().mean()))
    return {'sharp_test':rows,'mean_psnr':float(np.mean([r['psnr'] for r in rows])), 'mean_ssim':float(np.mean([r['ssim'] for r in rows])),
            'train_mse_double_order':float(np.mean(losses)),'quadrature_mse_calls_vs_double':float(np.mean(quad))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='dataset/evdeblurnerf_real/ev-deblurnerf_cdavis');p.add_argument('--scene',default='blurbatteries');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--points',type=int,default=8192);p.add_argument('--warmup',type=int,default=1200);p.add_argument('--steps',type=int,default=600);p.add_argument('--calls',type=int,default=8);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed)
    d=Dataset(a.root,a.scene);result={'args':vars(a),'dataset':d.manifest(),'protocol':'shared train-only warmup; all arms same order and steps; no test selection; RGB frame loss only; all XYZ trainable; static slider scene',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(result,indent=2))
    raw=initialize(d,a.seed,a.points)
    raw,info=train(d,raw,'scale-aware','gauss',a.warmup,a.calls,a.seed,warmup=True);result['warmup_training']=info
    torch.save(raw,out/'shared_initial.pt')
    arms={}
    for name,mode,method in [('original','legacy','partition'),('modified','scale-aware','partition'),('gauss','scale-aware','gauss')]:
        model,info=train(d,raw,mode,method,a.steps,a.calls,a.seed+123)
        torch.save(model,out/(name+'.pt'));arms[name]=info
    # Test pixels are never used by initialization or any train call.
    result['initial']=evaluate(d,raw,out/'initial',a.calls)
    for name in arms:
        model=torch.load(out/(name+'.pt'),weights_only=True);arms[name]['metrics']=evaluate(d,model,out/name,a.calls)
    result['arms']=arms
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v['metrics'] for k,v in arms.items()}),flush=True)


if __name__=='__main__':main()
