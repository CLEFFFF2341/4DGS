"""Measured-shutter dynamic stereo reconstruction on real DSEC RGB clips.

All models use a camera-rig coordinate gauge: the calibrated stereo cameras are
fixed, and both ego-induced scene motion and independently moving objects are
represented by the 4D Gaussians. No world trajectory is claimed or fabricated.
Held-out timestamps are never used by stereo initialization or training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from torchmetrics.functional.image import structural_similarity_index_measure

from experiments.n1b.core import Camera, render_cuda
from experiments.n1b_v2.joint_optimization import JointParameters
from experiments.n1b_v2.real_cdavis import schedule


class DynamicData:
    def __init__(self,root,start,count=24,width=360):
        self.root=Path(root);self.start=start;self.frames=list(range(start,start+count));self.w=width;self.h=width*3//4
        c=yaml.safe_load((self.root/'cam_to_cam.yaml').read_text())
        transforms=[]
        for i in (1,2):
            R=np.eye(4);R[:3,:3]=c['extrinsics'][f'R_rect{i}'];transforms.append(R)
        right=transforms[1]@np.array(c['extrinsics']['T_21'])@np.linalg.inv(transforms[0])
        self.cameras=[]
        for i,T in zip((1,2),(np.eye(4),right)):
            fx,fy,cx,cy=c['intrinsics'][f'camRect{i}']['camera_matrix'];sx=width/1440;sy=self.h/1080
            W=torch.tensor(T,device='cuda',dtype=torch.float32);pos=torch.linalg.inv(W)[:3,3]
            self.cameras.append(Camera(W,pos,self.w,self.h,fx*sx,fy*sy,cx*sx,cy*sy,.05,300.))
        times=[np.loadtxt(self.root/f'exposure_{s}.txt',delimiter=',')[self.frames] for s in ('left','right')]
        self.origin=float(min(t[0,0] for t in times));self.span=float(max(t[-1,1] for t in times)-self.origin)/1e6
        self.bounds=np.stack([(t-self.origin)/1e6/self.span for t in times])
        assert np.all(self.bounds[:,:,1]>self.bounds[:,:,0])
        self.train_frames=[i for i in range(count) if i%4!=2]
        self.test_frames=[i for i in range(count) if i%4==2]
        self.train=[(s,i) for i in self.train_frames for s in (0,1)]
        self.test=[(s,i) for i in self.test_frames for s in (0,1)]
        self.cpu=[];self.images=[]
        for side in ('left','right'):
            images=[np.asarray(Image.open(self.root/side/f'{i:06d}.png').convert('RGB').resize((self.w,self.h),Image.Resampling.LANCZOS)).copy() for i in self.frames]
            self.cpu.append(images);self.images.append([torch.tensor(im,device='cuda').permute(2,0,1).float()/255 for im in images])
        self.bg=torch.zeros(3,device='cuda')
        # Fixed evaluation ROIs declared before training: truck in the center,
        # passing tram in the left foreground. These are rectangles, not GT masks.
        yy,xx=torch.meshgrid(torch.arange(self.h,device='cuda')/self.h,torch.arange(self.w,device='cuda')/self.w,indexing='ij')
        self.roi=((xx>.40)&(xx<.66)&(yy>.32)&(yy<.72))|((xx<.34)&(yy>.40)&(yy<.86))

    def manifest(self):
        return {'sequence':self.root.name,'global_frames':self.frames,'train_frame_indices':[self.frames[i] for i in self.train_frames],
                'test_frame_indices':[self.frames[i] for i in self.test_frames],'origin_us':self.origin,'time_span_seconds':self.span,
                'measured_exposure_seconds':(self.bounds[:,:,1]-self.bounds[:,:,0]).__mul__(self.span).tolist(),
                'normalized_exposure_bounds':self.bounds.tolist(),'resolution':[self.w,self.h],
                'w2c':[c.w2c.cpu().tolist() for c in self.cameras],
                'coordinate_gauge':'calibrated camera rig; scene includes ego-induced and independent motion',
                'calibration_sha256':hashlib.sha256((self.root/'cam_to_cam.yaml').read_bytes()).hexdigest(),
                'download_manifest_sha256':hashlib.sha256((self.root/'download_manifest.json').read_bytes()).hexdigest()}


@torch.no_grad()
def initialize(data,seed,n):
    """Training-only stereo depths; test views/times are excluded completely."""
    rng=np.random.default_rng(seed);pools=[]
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=64,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2)
    baseline=float(torch.linalg.norm(data.cameras[1].position));cam=data.cameras[0]
    for i in data.train_frames:
        left,right=data.cpu[0][i],data.cpu[1][i]
        disp=matcher.compute(cv2.cvtColor(left,cv2.COLOR_RGB2GRAY),cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)).astype(np.float32)/16
        valid=(disp>1)&(disp<63);y,x=np.nonzero(valid)
        if len(x)<10:raise ValueError('Insufficient training stereo matches')
        chosen=rng.choice(len(x),max(n//len(data.train_frames)+1,1),replace=True);x=x[chosen];y=y[chosen]
        z=cam.fx*baseline/disp[y,x]
        xyz=np.stack(((x+.5-cam.cx)*z/cam.fx,(y+.5-cam.cy)*z/cam.fy,z),-1)
        rgb=left[y,x].astype(np.float32)/255
        t=np.full((len(x),1),data.bounds[0,i].mean(),np.float32)
        pools.append((xyz,rgb,t,z))
    xyz,rgb,t,z=[np.concatenate([p[j] for p in pools])[:n] for j in range(4)]
    identity=torch.zeros(n,4,device='cuda');identity[:,0]=1
    tensor=lambda v:torch.tensor(v,device='cuda',dtype=torch.float32)
    return {'spatial_means':tensor(xyz),'temporal_means':tensor(t),
            'spatial_scales':tensor(np.repeat(np.log(z[:,None]/cam.fx*2.5),3,axis=1)),
            'temporal_scales':torch.full((n,1),math.log(.10),device='cuda'),
            'left_isoclinic_rotations':identity.clone(),'right_isoclinic_rotations':identity.clone(),
            'opacities':torch.full((n,1),math.log(.2/.8),device='cuda'),
            'sh_coefficients_0':tensor(((rgb-.5)/.28209479177387814)[:,None]),
            'sh_coefficients_rest':torch.zeros(n,15,3,device='cuda')}


def predict(raw,data,s,i,calls,method):
    lo,hi=data.bounds[s,i];nodes,mass,equal=schedule(raw,float(lo),float(hi),calls,method)
    pred=data.bg[:,None,None].expand(3,data.h,data.w)*mass
    for t,w in nodes:pred=pred+float(w)*render_cuda(raw,data.cameras[s],float(t),data.bg)
    if not nodes:pred=pred+raw['spatial_means'].sum()*0
    return pred,equal


def train(data,initial,mode,method,steps,calls,seed,warmup=False):
    joint=JointParameters(initial,mode=mode);opt=joint.optimizer(lr=.01 if mode=='legacy' else .05,spatial_lr=.01)
    nuisance=[('sh_coefficients_0',.01),('sh_coefficients_rest',.0005),('opacities',.01),('spatial_scales',.003),
              ('temporal_means',.0005),('left_isoclinic_rotations',.0002),('right_isoclinic_rotations',.0002)]
    for key,lr in nuisance:
        joint.fixed[key].requires_grad_();opt.add_param_group({'params':[joint.fixed[key]],'lr':lr})
    if warmup:joint.temporal.requires_grad_(False)
    rng=np.random.default_rng(seed);order=[]
    while len(order)<steps:order.extend([data.train[i] for i in rng.permutation(len(data.train))])
    torch.cuda.synchronize();start=time.perf_counter();losses=[];same=0
    for step,(s,i) in enumerate(order[:steps]):
        opt.zero_grad(set_to_none=True);pred,equal=predict(joint.raw(),data,s,i,calls,method);same+=int(equal)
        loss=(pred-data.images[s][i]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();opt.step();losses.append(float(loss.detach()))
        if (step+1)%200==0:print(json.dumps({'stage':'warmup' if warmup else mode+'-'+method,'step':step+1,'loss':losses[-1],'seconds':time.perf_counter()-start}),flush=True)
    torch.cuda.synchronize()
    return {k:v.detach().clone() for k,v in joint.raw().items()},{'steps':steps,'calls':calls,'seconds':time.perf_counter()-start,'global_gauss_equal_steps':same,'loss_first':losses[0],'loss_last':losses[-1]}


@torch.no_grad()
def evaluate(raw,data,folder,calls=32):
    folder.mkdir(parents=True,exist_ok=True);rows=[];quad=[]
    for s,i in data.test:
        a,_=predict(raw,data,s,i,calls//2,'gauss');b,_=predict(raw,data,s,i,calls,'gauss')
        quad.append(float((a-b).square().mean()));pred=b.clamp(0,1);ref=data.images[s][i]
        mse=float((pred-ref).square().mean());roi_mse=float((pred[:,data.roi]-ref[:,data.roi]).square().mean())
        rows.append({'side':s,'frame':data.frames[i],'psnr':-10*math.log10(mse),'roi_psnr':-10*math.log10(roi_mse),
                     'ssim':float(structural_similarity_index_measure(pred[None],ref[None],data_range=1.))})
        im=(pred.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8);Image.fromarray(im).save(folder/f'{s}_{data.frames[i]:06d}.png')
    return {'views':rows,'mean_psnr':float(np.mean([r['psnr'] for r in rows])),'mean_ssim':float(np.mean([r['ssim'] for r in rows])),
            'mean_roi_psnr':float(np.mean([r['roi_psnr'] for r in rows])),'quadrature_mse_16_vs_32':float(np.mean(quad))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='dataset/dsec_dynamic/zurich_city_04_c');p.add_argument('--start',type=int,default=120);p.add_argument('--count',type=int,default=24)
    p.add_argument('--width',type=int,default=360);p.add_argument('--points',type=int,default=32768);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--warmup',type=int,default=1200);p.add_argument('--steps',type=int,default=800);p.add_argument('--calls',type=int,default=8);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);d=DynamicData(a.root,a.start,a.count,a.width)
    result={'args':vars(a),'dataset':d.manifest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol':'same train-only stereo initialization, shared warmup, same train order/steps/render budget/nuisance optimization; held-out timestamps; Gauss32 evaluation'}
    (out/'protocol.json').write_text(json.dumps(result,indent=2));raw=initialize(d,a.seed,a.points)
    raw,result['warmup']=train(d,raw,'scale-aware','gauss',a.warmup,a.calls,a.seed,warmup=True);torch.save(raw,out/'shared_initial.pt')
    result['arms']={}
    for name,mode,method in [('original','legacy','partition'),('modified','scale-aware','partition'),('gauss','scale-aware','gauss')]:
        model,info=train(d,raw,mode,method,a.steps,a.calls,a.seed+123)
        torch.save(model,out/(name+'.pt'));result['arms'][name]=info
    result['initial']=evaluate(raw,d,out/'initial')
    for name,info in result['arms'].items():info['metrics']=evaluate(torch.load(out/(name+'.pt'),weights_only=True),d,out/name)
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['metrics'] for k,v in result['arms'].items()}),flush=True)


if __name__=='__main__':main()
