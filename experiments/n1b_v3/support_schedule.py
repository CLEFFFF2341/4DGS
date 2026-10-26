"""Bounded support-edge partition proposals; no all-or-nothing overflow fallback."""
import math
import numpy as np
import torch
from experiments.n1b.core import project
from experiments.n1b_v2.exposure import gauss_rule


@torch.no_grad()
def schedule(raw, camera, lo, hi, budget, method='weighted_edges'):
    if method not in ('weighted_edges', 'top_edges') or budget < 2 or hi <= lo:
        raise ValueError('invalid proposal configuration')
    center, motion, cov, _, _, _, variance, valid = project(raw, camera, (lo+hi)/2)
    radius = 3*cov.diagonal(dim1=-2, dim2=-1).sqrt() + motion.abs()*(hi-lo)/2
    valid &= ((center+radius) > 0).all(-1)
    valid &= (center[:, 0]-radius[:, 0] < camera.width) & (center[:, 1]-radius[:, 1] < camera.height)
    area = (cov[:, 0, 0]*cov[:, 1, 1]-cov[:, 0, 1].square()).clamp_min(0).sqrt()
    # Approximate screen-space contribution, NOT exact occlusion/visibility.
    importance = raw['opacities'][:, 0].sigmoid() * area.clamp(max=camera.width*camera.height) * valid
    time_radius = math.sqrt(-2*math.log(.05))*variance.sqrt()
    time_center = raw['temporal_means'][:, 0]
    ts = torch.cat((time_center-time_radius, time_center+time_radius))
    score = torch.cat((importance, importance))
    eligible = (ts > lo) & (ts < hi) & torch.isfinite(score) & (score > 0)
    ts, score = ts[eligible].cpu().numpy(), score[eligible].cpu().numpy()
    cuts = [lo, hi]
    max_internal = min(3, budget//2-1)
    if len(ts) and max_internal > 0:
        order = np.argsort(ts); ts, score = ts[order], score[order]
        unique, inverse = np.unique(ts, return_inverse=True)
        merged = np.bincount(inverse, weights=score)
        if method == 'top_edges':
            cuts += unique[np.argsort(merged)[-max_internal:]].tolist()
        else:
            cumulative = np.cumsum(merged)
            for fraction in np.linspace(0, 1, max_internal+2)[1:-1]:
                cuts.append(float(unique[np.searchsorted(cumulative, fraction*cumulative[-1])]))
    cuts = sorted(set(cuts)); intervals = list(zip(cuts[:-1], cuts[1:]))
    counts = [1]*len(intervals)
    for _ in range(budget-len(intervals)):
        j = max(range(len(intervals)), key=lambda j: (intervals[j][1]-intervals[j][0])/counts[j])
        counts[j] += 1
    output = []
    for (a, b), n in zip(intervals, counts):
        x, w = gauss_rule(n)
        output.extend(zip((a+b)/2+x*(b-a)/2, w*(b-a)/(2*(hi-lo))))
    return output, {'considered_edges': len(ts), 'cuts': cuts, 'counts': counts}
