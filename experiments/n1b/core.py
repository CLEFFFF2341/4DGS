"""Parameter-derived Gaussian/pair control variate for a uniform shutter.

Discrete lists are detached; all selected raw-parameter paths stay differentiable.
The proxy is deliberately unclipped, including when degree-two composition is negative.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Camera:
    w2c: torch.Tensor
    position: torch.Tensor
    width: int = 32
    height: int = 32
    fx: float = 40.
    fy: float = 40.
    cx: float = 16.
    cy: float = 16.
    near: float = .1
    far: float = 100.


def rotation(left, right):
    a, b, c, d = F.normalize(left, dim=-1).unbind(-1)
    p, q, r, s = F.normalize(right, dim=-1).unbind(-1)
    L = torch.stack((a,-b,-c,-d,b,a,-d,c,c,d,a,-b,d,-c,b,a), -1).reshape(*a.shape,4,4)
    R = torch.stack((p,-q,-r,-s,q,p,s,-r,r,-s,p,q,s,r,-q,p), -1).reshape(*p.shape,4,4)
    return L @ R


def colors(raw, camera, bases):
    if bases not in (1,4,9,16):
        raise ValueError('active SH bases must be 1, 4, 9 or 16')
    sh = torch.cat((raw['sh_coefficients_0'], raw['sh_coefficients_rest']), -2)
    x,y,z = F.normalize(raw['spatial_means'] - camera.position, dim=-1).unbind(-1)
    terms = [torch.ones_like(x)*.28209479177387814]
    if bases > 1:
        terms += [-.48860251190291987*y, .48860251190291987*z, -.48860251190291987*x]
    if bases > 4:
        terms += [1.0925484305920792*x*y, -1.0925484305920792*y*z,
                  .94617469575756*z*z-.31539156525252, -1.0925484305920792*x*z,
                  .5462742152960396*(x*x-y*y)]
    if bases > 9:
        terms += [y*(.5900435899266435*y*y-1.7701307697799304*x*x),
                  2.890611442640554*x*y*z, y*(.4570457994644657-2.285228997322329*z*z),
                  z*(1.865881662950577*z*z-1.1195289977703462),
                  x*(.4570457994644657-2.285228997322329*z*z),
                  1.445305721320277*z*(x*x-y*y),
                  x*(1.7701307697799304*y*y-.5900435899266435*x*x)]
    return (.5 + (torch.stack(terms,-1)[...,None]*sh[...,:bases,:]).sum(-2)).clamp_min(0)


def project(raw, camera, t0):
    left,right=raw['left_isoclinic_rotations'],raw['right_isoclinic_rotations']
    valid_rotation=(left.square().sum(-1)>=1e-8)&(right.square().sum(-1)>=1e-8)
    identity=left.new_tensor([1,0,0,0])
    R = rotation(torch.where(valid_rotation[...,None],left,identity),
                 torch.where(valid_rotation[...,None],right,identity))
    scales = torch.cat((raw['spatial_scales'], raw['temporal_scales']),-1)
    cov = (R * (2*scales).exp()[...,None,:]) @ R.transpose(-1,-2)
    s2 = cov[...,3,3]
    v = cov[...,:3,3]/s2[...,None]
    C = cov[...,:3,:3] - cov[...,:3,3,None]*v[...,None,:]
    dt = t0-raw['temporal_means'][...,0]
    world = raw['spatial_means']+v*dt[...,None]
    W = camera.w2c[:3,:3].to(world)
    xyz = world @ W.T + camera.w2c[:3,3].to(world)
    valid = torch.isfinite(xyz).all(-1) & (xyz[...,2] >= camera.near)
    valid &= valid_rotation
    # Invalid midpoints cannot enter the list; safe arithmetic avoids invalid J/inverses.
    xyz = torch.where(valid[...,None], xyz, xyz.new_tensor([0,0,1]))
    x,y,z = xyz.unbind(-1)
    zero = torch.zeros_like(x)
    J = torch.stack((camera.fx/z,zero,-camera.fx*x/z.square(),
                     zero,camera.fy/z,-camera.fy*y/z.square()),-1).reshape(*x.shape,2,3)
    JW = J @ W
    S = JW @ C @ JW.transpose(-1,-2) + .3*torch.eye(2,device=x.device,dtype=x.dtype)
    Q = torch.linalg.inv(S)
    center = torch.stack((camera.fx*x/z+camera.cx,camera.fy*y/z+camera.cy),-1)
    u = (JW @ v[...,None])[...,0]
    return center,u,S,Q,z,dt,s2,valid


@torch.no_grad()
def select_ids(raw, camera, pixels, t0, duration, cap=16, primitive_chunk=1024, pixel_chunk=256):
    """Bounded N_chunk x P_chunk allocation; depth then ID ordering, no tail clipping."""
    if duration <= 0 or cap < 1 or primitive_chunk < 1 or pixel_chunk < 1:
        raise ValueError('positive shutter, cap and chunks required')
    n = len(raw['spatial_means'])
    if n == 0:
        raise ValueError('empty raw model')
    selected=[]; counts=[]; invalid=0
    for start in range(0,len(pixels),pixel_chunk):
        pp=pixels[start:start+pixel_chunk]
        ids=torch.full((len(pp),cap),-1,dtype=torch.long,device=pp.device)
        depths=torch.full((len(pp),cap),float('inf'),dtype=pp.dtype,device=pp.device)
        count=torch.zeros(len(pp),dtype=torch.long,device=pp.device)
        for j in range(0,n,primitive_chunk):
            rr={k:v[j:j+primitive_chunk] for k,v in raw.items()}
            center,u,S,_,z,_,_,valid=project(rr,camera,t0)
            if start==0: invalid += int((~valid).sum())
            radius=math.sqrt(2*math.log(255))*S.diagonal(dim1=-2,dim2=-1).sqrt()
            extent=radius+u.abs()*duration/2
            hit=((pp[:,None]-center).abs()<=extent).all(-1)&valid
            count += hit.sum(-1)
            candidate=torch.arange(j,j+len(z),device=pp.device).expand(len(pp),-1)
            dd=torch.where(hit,z,float('inf'))
            all_ids=torch.cat((ids,candidate),-1)
            all_depth=torch.cat((depths,dd),-1)
            # Earlier chunks have smaller IDs; stable sort preserves tie order.
            take=all_depth.argsort(dim=-1,stable=True)[:,:cap]
            depths=all_depth.gather(1,take); ids=all_ids.gather(1,take)
            ids=torch.where(torch.isfinite(depths),ids,-1)
        selected.append(ids); counts.append(count)
    ids=torch.cat(selected); count=torch.cat(counts)
    return ids, {'overflow_pixels':int((count>cap).sum()),'max_candidates':int(count.max()),
                 'invalid_midpoints':invalid,'selected_slots':int((ids>=0).sum())}


def gaussian_average(a,b,d,duration):
    """Stable analytic mean of exp(a+b*h-d*h^2/2), for d>=0.

    Float64, log-CDF subtraction in the tails, convergent polynomial near d=0.
    The small-exponent branch retains derivatives at d=0 (no detached constant).
    """
    if duration <= 0:
        raise ValueError('duration must be positive')
    a,b,d=torch.broadcast_tensors(a.double(),b.double(),d.double())
    if bool((d<0).any()):
        raise ValueError('negative precision')
    if bool(((d==0)&(b!=0)).any()):
        raise ValueError('zero precision requires zero slope for a Gaussian pulse')
    small=(b.abs()*duration + d*duration**2 < 1e-3)
    # Taylor integration through fourth order in b*h-d*h*h/2.
    series=torch.ones_like(a)
    for n in range(1,5):
        for k in range(n+1):
            power=n+k
            if power%2==0:
                moment=(duration/2)**power/(power+1)
                series=series+math.comb(n,k)/math.factorial(n)*b**(n-k)*(-d/2)**k*moment
    # Safe dummy inputs on the series branch prevent 0*nan in autograd.
    dd=torch.where(small,torch.ones_like(d),d).clamp_min(1e-300)
    bb=torch.where(small,torch.zeros_like(b),b)
    root=dd.sqrt(); mu=bb/dd
    lo=(-duration/2-mu)*root; hi=(duration/2-mu)*root
    # Reflect positive tails: subtract CDFs on the negative half line.
    l=torch.where(lo>0,-hi,lo); h=torch.where(lo>0,-lo,hi)
    lh=torch.special.log_ndtr(h); ll=torch.special.log_ndtr(l)
    delta=(ll-lh).clamp_max(-torch.finfo(a.dtype).eps)
    logmass=lh+torch.log(-torch.expm1(delta))
    logvalue=a+bb.square()/(2*dd)+.5*(math.log(2*math.pi)-dd.log())+logmass-math.log(duration)
    return torch.where(small,a.exp()*series,logvalue.exp())


@dataclass
class Proxy:
    a: torch.Tensor
    b: torch.Tensor
    d: torch.Tensor
    color: torch.Tensor
    mask: torch.Tensor
    background: torch.Tensor
    duration: float

    def alpha(self,h):
        return (self.a+self.b*h-.5*self.d*h*h).exp()*self.mask

    def value(self,h,pair=True):
        alpha=self.alpha(h)
        weight=alpha
        if pair:
            weight=alpha*(1-(alpha.cumsum(-1)-alpha))
        return self.background+((self.color-self.background)*weight[...,None]).sum(-2)

    def integral(self,pair=True):
        A=gaussian_average(self.a,self.b,self.d,self.duration)*self.mask
        value=self.background+((self.color-self.background)*A[...,None]).sum(-2)
        if pair:
            # One back layer at a time, no P*M*M*RGB allocation.
            for i in range(1,self.a.shape[-1]):
                pairA=gaussian_average(self.a[...,:i]+self.a[...,i,None],
                    self.b[...,:i]+self.b[...,i,None],self.d[...,:i]+self.d[...,i,None],self.duration)
                pairA=pairA*self.mask[...,:i]*self.mask[...,i,None]
                value=value-(self.color[...,i,:]-self.background)*pairA.sum(-1)[...,None]
        return value

    def full_local(self,h):
        alpha=self.alpha(h)
        trans=torch.cat((torch.ones_like(alpha[...,:1]),(1-alpha).cumprod(-1)[...,:-1]),-1)
        return self.background+((self.color-self.background)*(alpha*trans)[...,None]).sum(-2)


def build_proxy(raw,camera,pixels,ids,t0,duration,background,bases=16):
    """Second pass builds graphs only for selected parameters (including repeated IDs)."""
    rr={k:v[ids.clamp_min(0)].double() for k,v in raw.items()}
    center,u,_,Q,_,dt,s2,_=project(rr,camera,t0)
    e=pixels.double()[:,None]-center
    a=F.logsigmoid(rr['opacities'][...,0])-.5*((e*(Q@e[...,None])[...,0]).sum(-1)+dt.square()/s2)
    b=(e*(Q@u[...,None])[...,0]).sum(-1)-dt/s2
    d=(u*(Q@u[...,None])[...,0]).sum(-1)+1/s2
    return Proxy(a,b,d,colors(rr,camera,bases),ids>=0,background.double(),duration)


def estimate(render,proxy,times,t0,pair=True,weights=None):
    """times shared by all pixels; weights=w/q for a fixed positive proposal."""
    if len(times)==0:
        raise ValueError('at least one residual sample is required')
    residual=[]
    for i,t in enumerate(times):
        weight=1 if weights is None else weights[i]
        residual.append(weight*(render(float(t))-proxy.value(t-t0,pair)))
    return proxy.integral(pair)+torch.stack(residual).mean(0)


def independent_mse(first,second,target):
    """Caller must supply independent residual samples; not an L1/SSIM estimator."""
    return ((first-target)*(second-target)).mean()


def render_cuda(raw,camera,timestamp,background,bases=16):
    """Exactly the unclamped diff_rasterize training path; no optimizer/statistics."""
    from FasterGS4DCudaBackend.torch_bindings.rasterization import diff_rasterize, RasterizerSettings
    settings=RasterizerSettings(camera.w2c.float().contiguous(),camera.position.float().contiguous(),
        background.float().contiguous(),bases,camera.width,camera.height,camera.fx,camera.fy,
        camera.cx,camera.cy,camera.near,camera.far,float(timestamp))
    return diff_rasterize(**{k:v.float().contiguous() for k,v in raw.items()},
        densification_info=torch.empty(0,device=background.device),rasterizer_settings=settings)
