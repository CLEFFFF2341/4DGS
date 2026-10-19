"""Detached temporal-support partitions, native first-order gradients only.

Deterministic quadrature is biased. Target gradients are expected backend branch
AD, as in the archived reference, NOT derivatives of moving culling boundaries.
No claim of unbiasedness or of differentiating a moving quadrature formula.
"""
import math
import numpy as np
import torch
from experiments.n1b.core import rotation


@torch.no_grad()
def nodes(raw, budget, lo=.3, hi=.7, allocation='equal'):
    R = rotation(raw['left_isoclinic_rotations'], raw['right_isoclinic_rotations'])
    variance = (2*torch.cat((raw['spatial_scales'], raw['temporal_scales']), -1)).exp()
    sigma = (R[:, 3].square()*variance).sum(-1).sqrt()
    # Native temporal marginal < .05 is culled, independent of screen position.
    radius = math.sqrt(-2*math.log(.05))*sigma
    center = raw['temporal_means'][:, 0]
    windows = [(max(lo, a), min(hi, b)) for a, b in zip((center-radius).cpu().tolist(), (center+radius).cpu().tolist()) if a < hi and b > lo]
    cuts = sorted(set([lo, hi]+[t for ab in windows for t in ab]))
    intervals = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if any(c < (a+b)/2 < d for c, d in windows)]
    if len(intervals) > budget//2:
        # Fail closed to global quadrature rather than silently discard support.
        intervals = [(lo, hi)]
    if not intervals:
        return [], 1., {'intervals': 0, 'calls': 0}
    count = [budget//len(intervals)+(i < budget%len(intervals)) for i in range(len(intervals))]
    if allocation == 'balanced':
        # Equal calls per interval waste work on tiny intervals formed by nearly
        # coincident support boundaries. Allocate by width in local pulse units.
        scales = sigma.cpu().tolist()
        centers = center.cpu().tolist()
        scores = []
        for a, b in intervals:
            mid = (a+b)/2
            active = [s for c, s in zip(centers, scales) if abs(mid-c) <= math.sqrt(-2*math.log(.05))*s]
            scores.append((b-a)/max(min(active, default=hi-lo), 1e-15))
        count = [1]*len(intervals)
        for _ in range(budget-len(intervals)):
            i = max(range(len(intervals)), key=lambda j: scores[j]/count[j])
            count[i] += 1
    output = []
    for (a, b), k in zip(intervals, count):
        x, w = np.polynomial.legendre.leggauss(k)
        output.extend(zip(((a+b)/2+x*(b-a)/2).tolist(), (w*(b-a)/(2*(hi-lo))).tolist()))
    omitted = 1-sum(b-a for a, b in intervals)/(hi-lo)
    return output, max(0., omitted), {'intervals': len(intervals), 'calls': len(output)}


def estimate(raw, cam, bg, weight, calls, gradient, allocation='equal'):
    from experiments.n1b.narrow_benchmark import render, vjp
    ts, background_mass, diagnostic = nodes(raw, calls, allocation=allocation)
    result = torch.zeros(sum(v.numel() for v in raw.values()), device=bg.device, dtype=torch.double) if gradient else bg.double().expand(64, 3)*background_mass
    for t, w in ts:
        value = render(raw, cam, bg, t)
        result = result+w*(vjp(value, raw, weight) if gradient else value)
    return result.detach(), diagnostic
