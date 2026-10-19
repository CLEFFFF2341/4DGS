"""Known-shutter, linear-RGB synthetic fixed-model fixtures (native raw65)."""
import math
import torch
from .core import Camera


def scene(name,device='cuda'):
    n={'single':1,'double':2,'triple':3,'static_layers':9,'dense':16,
       'static':4,'perspective':2,'overflow':24}[name]
    def tensor(x): return torch.tensor(x,device=device,dtype=torch.float32)
    raw={
        'spatial_means':torch.zeros(n,3,device=device),
        'temporal_means':torch.full((n,1),.5,device=device),
        'spatial_scales':torch.full((n,3),math.log(.065),device=device),
        'temporal_scales':torch.full((n,1),math.log(.08),device=device),
        'left_isoclinic_rotations':tensor([[1,0,0,0]]*n),
        'right_isoclinic_rotations':tensor([[1,0,0,0]]*n),
        'opacities':torch.full((n,1),math.log(.8/.2),device=device),
        'sh_coefficients_0':torch.zeros(n,1,3,device=device),
        'sh_coefficients_rest':torch.zeros(n,15,3,device=device),
    }
    raw['spatial_means'][:,2]=torch.linspace(3,3.3,n,device=device)
    raw['temporal_means'][:,0]=torch.linspace(.43,.57,n,device=device)
    raw['spatial_scales'][:,1]=math.log(.45)
    # x/time rotation and unequal scales create moving thin stripes.
    raw['temporal_scales'][:]=math.log(.4)
    angle=.4
    raw['left_isoclinic_rotations'][:,0]=math.cos(angle)
    raw['left_isoclinic_rotations'][:,3]=math.sin(angle)
    if n>1: raw['left_isoclinic_rotations'][1,3]*=-1
    palette=tensor([[.9,.2,.1],[.1,.7,.9],[.2,.9,.2]])
    raw['sh_coefficients_0'][:,0]=(palette[torch.arange(n,device=device)%3]-.5)/.28209479177387814
    if name in ('dense','overflow'):
        raw['spatial_scales'][:,0]=math.log(.3)
        raw['temporal_scales'][:]=math.log(.12)
        raw['left_isoclinic_rotations'][:]=tensor([1,0,0,0])
        raw['temporal_means'][:]=.46
    if name in ('static','static_layers'):
        end=n if name=='static' else n-1
        raw['left_isoclinic_rotations'][:end]=tensor([1,0,0,0])
        raw['temporal_scales'][:end]=math.log(1e6)
        raw['spatial_scales'][:end,0]=math.log(.3)
        raw['temporal_means'][:end]=.5
    if name=='perspective':
        raw['left_isoclinic_rotations'][:]=tensor([math.cos(.6),math.sin(.6),0,0])
        raw['temporal_scales'][:]=math.log(1.)
        raw['spatial_means'][:,0]=tensor([.1,-.1])
    camera=Camera(torch.eye(4,device=device),torch.zeros(3,device=device))
    yy,xx=torch.meshgrid(torch.arange(12,20,device=device),torch.arange(12,20,device=device),indexing='ij')
    # CUDA evaluates pixel centers at integer + 0.5.
    pixels=torch.stack((xx.flatten()+.5,yy.flatten()+.5),-1)
    return raw,camera,pixels,tensor([.03,.05,.07])
