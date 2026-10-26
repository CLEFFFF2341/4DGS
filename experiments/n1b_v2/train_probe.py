"""Known-exposure synthetic fitting; never touches the real checkpoint.

Update only log temporal scales and spatial means, common Adam and 32 calls/step.
Equal 3-second end-to-end local training budgets, with native reference evaluation
outside training time. Last over-budget update is rolled back. No GT error stopping.
"""
import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from experiments.n1b.narrow_benchmark import fixture, reference, proxy
from experiments.n1b.benchmark import Proposal, sync
from .partition import nodes
from experiments.n1b.core import render_cuda


def render_pixels(raw, cam, pix, bg, t):
    """Select integer-center pixels without hard-coding the central 8x8 ROI."""
    image = render_cuda(raw, cam, t, bg)
    xy = pix.long()
    return image[:, xy[:, 1], xy[:, 0]].T.double()


def prediction(method, raw, cam, pix, bg, calls=32):
    if method == 'partition':
        ts, background, _ = nodes(raw, calls, allocation='balanced')
    elif method == 'gauss':
        x, w = np.polynomial.legendre.leggauss(calls)
        ts = list(zip(.5+.2*x, w/2)); background = 0
    elif method == 'stratified':
        ts = [(t, 1/calls) for t in (.3+.4*(torch.arange(calls, device='cuda')+torch.rand(calls, device='cuda'))/calls).tolist()]
        background = 0
    else:
        with torch.no_grad():
            h, w = Proposal(proxy(raw, cam, pix, bg)).sample(calls)
        ts = list(zip((h+.5).tolist(), (w/calls).tolist())); background = 0
    result = bg.double().expand(len(pix), 3)*background
    for t, w in ts:
        result = result+float(w)*render_pixels(raw, cam, pix, bg, float(t))
    return result


def main(seed, only_method=None, calls=32, views=1, temporal_only=False):
    torch.set_num_threads(4)
    rng = np.random.default_rng(87000+seed)
    sigma = float(np.exp(rng.uniform(np.log(.001), np.log(.004))))
    truth, cam, pix, bg = fixture(sigma, float(rng.uniform(-.01, .01)), bool(seed % 2))
    truth = {k: v.requires_grad_() for k, v in truth.items()}
    W = cam.w2c.clone(); W[0, 3] -= .025
    check_cam = replace(cam, w2c=W, position=cam.position+cam.position.new_tensor([.025, 0, 0]))
    weight = torch.ones(64, 3, device='cuda', dtype=torch.double)/192
    target, _, _ = reference(truth, cam, pix, bg, weight, 8)
    check_target, _, _ = reference(truth, check_cam, pix, bg, weight, 8)
    train_cameras, targets = [cam], [target]
    if views == 2:
        W2 = cam.w2c.clone(); W2[0, 3] += .03
        second_camera = replace(cam, w2c=W2, position=cam.position-cam.position.new_tensor([.03, 0, 0]))
        second_target, _, _ = reference(truth, second_camera, pix, bg, weight, 8)
        train_cameras.append(second_camera); targets.append(second_target)
    initial = {k: v.detach().clone() for k, v in truth.items()}
    initial['temporal_scales'] += math.log(1.7)
    if not temporal_only:
        initial['spatial_means'][:, 0] += .015
    # Shared framework startup is warmed equally before any method clock.
    dummy = torch.ones(1, device='cuda', requires_grad=True)
    warm = torch.optim.Adam([dummy]); dummy.sum().backward(); warm.step(); sync()
    out = dict(seed=seed, sigma=sigma, staggered=bool(seed % 2), budget_seconds=3., calls_per_step=calls,
               train_views=views, trainable=['temporal_scales'] if temporal_only else ['spatial_means', 'temporal_scales'], rows=[])
    suffix = '' if calls == 32 else f'_calls{calls}'
    suffix += '' if views == 1 else '_views2'
    suffix += '_temporal' if temporal_only else ''
    path = Path('experiments/n1b_v2/results')/f'train_probe_{seed}{suffix}.json'
    if only_method and path.exists():
        out['rows'] = [r for r in json.loads(path.read_text())['rows'] if r['method'] != only_method]
    for method in ([only_method] if only_method else ['gauss', 'stratified', 'pulse', 'stratified_independent', 'pulse_independent', 'partition']):
        torch.manual_seed(90000+seed)
        sync(); start = time.perf_counter()
        raw = {k: v.clone().requires_grad_() for k, v in initial.items()}
        optimizer = torch.optim.Adam([raw[k] for k in out['trainable']], lr=.01)
        steps = 0; last_loss = None; rollback = False
        while True:
            previous = {k: raw[k].detach().clone() for k in out['trainable']}
            for v in raw.values():
                v.grad = None
            current_camera, current_target = train_cameras[steps % views], targets[steps % views]
            if method.endswith('_independent'):
                base_method = method.removesuffix('_independent')
                first = prediction(base_method, raw, current_camera, pix, bg, calls//2)
                second = prediction(base_method, raw, current_camera, pix, bg, calls//2)
                loss = ((first-current_target)*(second-current_target)).mean()
            else:
                predicted = prediction(method, raw, current_camera, pix, bg, calls)
                loss = (predicted-current_target).square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite training loss')
            loss.backward(); optimizer.step()
            sync()
            if time.perf_counter()-start > 3.:
                with torch.no_grad():
                    for k, v in previous.items(): raw[k].copy_(v)
                rollback = True
                break
            steps += 1; last_loss = float(loss.detach())
        elapsed = time.perf_counter()-start
        image, _, _ = reference(raw, cam, pix, bg, weight, 8)
        check, _, _ = reference(raw, check_cam, pix, bg, weight, 8)
        train_mse = float((image-target).square().mean())
        check_mse = float((check-check_target).square().mean())
        row = dict(method=method, accepted_steps=steps, elapsed=elapsed, rollback=rollback,
            last_estimated_loss=last_loss, reference_train_mse=train_mse,
            reference_check_mse=check_mse, check_psnr=-10*math.log10(max(check_mse, 1e-30)),
            position_mse=float((raw['spatial_means']-truth['spatial_means']).detach().square().mean()),
            log_scale_mse=float((raw['temporal_scales']-truth['temporal_scales']).detach().square().mean()))
        out['rows'].append(row)
        path.write_text(json.dumps(out, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', choices=['gauss', 'stratified', 'pulse', 'stratified_independent', 'pulse_independent', 'partition'])
    parser.add_argument('--calls', type=int, choices=[16, 32, 64, 128], default=32)
    parser.add_argument('--views', type=int, choices=[1, 2], default=1)
    parser.add_argument('--temporal-only', action='store_true', help='separate causal probe: known fixed geometry, recover only temporal widths')
    args = parser.parse_args()
    main(args.seed, args.method, args.calls, args.views, args.temporal_only)
