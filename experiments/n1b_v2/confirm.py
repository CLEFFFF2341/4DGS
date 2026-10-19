"""Independent fixture/VJP confirmation, resumable, no accuracy-based tuning.

64 prescribed seeds vary phase, pulse width, peak staggering and VJP direction.
Methods share calibrated time caps with 40% headroom. Every overrun is retained.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.n1b.narrow_benchmark import fixture, reference, timed, calibrate
from .pilot import measure


def run(seed, allocation='equal'):
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed+23000)
    sigma = float(np.exp(rng.uniform(np.log(.0005), np.log(.02))))
    phase = float(rng.uniform(-.015, .015))
    staggered = bool(seed % 2)
    raw, cam, pix, bg = fixture(sigma, phase, staggered)
    # Independent temporal peaks make symmetry cancellation nontrivial.
    if staggered:
        raw['temporal_means'][:, 0] += torch.tensor(rng.uniform(-.008, .008, 8), device='cuda', dtype=torch.float32)
    raw = {k: v.requires_grad_() for k, v in raw.items()}
    generator = torch.Generator(device='cuda').manual_seed(23000+seed)
    weight = torch.randn(64, 3, generator=generator, device='cuda', dtype=torch.double)/192
    ref4, grad4, n4 = reference(raw, cam, pix, bg, weight, 4)
    ref, grad, n8 = reference(raw, cam, pix, bg, weight, 8)
    out = dict(seed=seed, allocation=allocation, sigma=sigma, phase=phase, staggered=staggered, reference_nodes=[n4, n8],
               reference_change=[float((ref-ref4).square().mean()), float((grad-grad4).square().mean())], rows=[])
    for gradient in [False, True]:
        for method in ['gauss', 'stratified', 'pulse', 'partition']:
            if method == 'partition':
                k = 16
                for _ in range(3):
                    _, dt, _ = measure(raw, cam, bg, weight, k, gradient, allocation)
                    k = max(2, min(256, int(k*.03/dt)))
            else:
                k = max(1, int(.75*calibrate(method, .05, raw, cam, pix, bg, weight, gradient)))
            torch.manual_seed(100000+seed)
            if method == 'partition':
                value, dt, detail = measure(raw, cam, bg, weight, k, gradient, allocation)
            else:
                value, dt = timed(method, k, raw, cam, pix, bg, weight, gradient)
                detail = {}
            target = grad if gradient else ref
            out['rows'].append(dict(method=method, gradient=gradient, calls=k, seconds=dt,
                within_budget=dt <= .05, mse=float((value-target).square().mean()),
                max_abs=float((value-target).abs().max()), **{k:v for k,v in detail.items() if k != 'calls'}))
    destination = Path('experiments/n1b_v2/results')/('confirmation' if allocation == 'equal' else 'confirmation_balanced'); destination.mkdir(parents=True, exist_ok=True)
    (destination/f'{seed:03d}.json').write_text(json.dumps(out, indent=2))
    ratios = []
    for gradient in [False, True]:
        rows = [r for r in out['rows'] if r['gradient'] == gradient]
        ours = next(r for r in rows if r['method'] == 'partition')
        best = min(r['mse'] for r in rows if r['method'] != 'partition')
        ratios.append(ours['mse']/max(best, 1e-30))
    print(json.dumps(dict(seed=seed, ratio_image_gradient=ratios, overrun=sum(not r['within_budget'] for r in out['rows']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--allocation', choices=['equal', 'balanced'], default='equal')
    args = parser.parse_args()
    for seed in range(args.start, args.start+args.count):
        run(seed, args.allocation)
