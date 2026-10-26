"""Known-exposure observations for joint fitting of small native models.

An exposure is explicit input metadata. A frame timestamp/fps is not substituted
for a shutter duration. The stationary reference is a diagnostic for fixtures,
not a general moving-scene integrator.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
import torch
from experiments.n1b.core import render_cuda, select_ids, build_proxy
from .partition import nodes


@dataclass(frozen=True)
class Exposure:
    lo: float
    hi: float

    def __post_init__(self):
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo >= self.hi:
            raise ValueError('known finite shutter bounds with lo < hi are required')

    @property
    def duration(self):
        return self.hi - self.lo


@lru_cache(maxsize=64)
def gauss_rule(order):
    if order < 1:
        raise ValueError('positive quadrature order required')
    return np.polynomial.legendre.leggauss(order)


def predict_exposure(raw, camera, background, exposure, calls=32, method='partition'):
    if calls < 1:
        raise ValueError('positive render budget required')
    if method == 'partition':
        schedule, mass, _ = nodes(raw, calls, exposure.lo, exposure.hi, allocation='balanced')
    elif method == 'gauss':
        x, w = gauss_rule(calls)
        schedule = zip((exposure.lo + exposure.hi)/2 + exposure.duration*x/2, w/2)
        mass = 0.
    else:
        raise ValueError('method must be partition or gauss')
    image = background.double()[:, None, None].expand(3, camera.height, camera.width) * mass
    count = 0
    for t, w in schedule:
        image = image + float(w) * render_cuda(raw, camera, float(t), background).double()
        count += 1
    if not count:
        # An inactive exposure has zero branch gradient, including during fitting.
        image = image + sum(v.sum() * 0 for v in raw.values())
    return image


@torch.no_grad()
def stationary_reference(raw, camera, background, exposure, order=8):
    # Validate the restricted reference instead of quietly accepting motion.
    from experiments.n1b.core import rotation
    rotation_matrix = rotation(raw['left_isoclinic_rotations'], raw['right_isoclinic_rotations'])
    identity = torch.eye(4, device=rotation_matrix.device, dtype=rotation_matrix.dtype)
    if not torch.allclose(rotation_matrix, identity.expand_as(rotation_matrix), atol=1e-6):
        raise ValueError('event reference requires stationary identity-rotation fixtures')
    yy, xx = torch.meshgrid(torch.arange(camera.height, device=background.device),
                           torch.arange(camera.width, device=background.device), indexing='ij')
    pixels = torch.stack((xx.flatten()+.5, yy.flatten()+.5), -1)
    midpoint = (exposure.lo + exposure.hi)/2
    ids, info = select_ids(raw, camera, pixels, midpoint, exposure.duration, cap=len(raw['spatial_means']))
    p = build_proxy(raw, camera, pixels, ids, midpoint, exposure.duration, background)
    center = p.b/p.d + midpoint
    peak = p.a + p.b.square()/(2*p.d)
    radius = (2*(peak-math.log(1/255)).clamp_min(0)/p.d).sqrt()
    cuts = [exposure.lo, exposure.hi]
    for value in (center-radius, center+radius):
        cuts += value[p.mask].flatten().tolist()
    tau = raw['temporal_means'][:, 0]
    temporal_radius = math.sqrt(-2*math.log(.05))*raw['temporal_scales'][:, 0].exp()
    cuts += (tau-temporal_radius).tolist() + (tau+temporal_radius).tolist()
    cuts = sorted(set(float(t) for t in cuts if exposure.lo <= t <= exposure.hi))
    x, w = gauss_rule(order)
    # CUDA takes a float timestamp: merge identical float32 nodes exactly.
    # This reduces redundant renders in tiny intervals without removing events.
    schedule = {}
    for a, b in zip(cuts[:-1], cuts[1:]):
        for t, weight in zip((a+b)/2 + x*(b-a)/2, w*(b-a)/(2*exposure.duration)):
            key = float(np.float32(t))
            schedule[key] = schedule.get(key, 0.) + float(weight)
    image = torch.zeros((3, camera.height, camera.width), device=background.device, dtype=torch.double)
    for t, weight in schedule.items():
        image.add_(render_cuda(raw, camera, t, background).double(), alpha=weight)
    return image
