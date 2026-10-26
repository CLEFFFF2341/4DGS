"""Information ablation: one long exposure versus known short-exposure frames.

This changes the available observations. It is not a same-data integrator win.
All arms fit XYZ and temporal width from wrong XYZ initialization, using the
same scale-aware optimizer. True parameters and check views only evaluate fits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from experiments.n1b.core import render_cuda
from experiments.n1b.narrow_benchmark import fixture
from experiments.n1b.benchmark import sync
from .joint_optimization import JointParameters
from .joint_probe import camera_at, parameter_errors
from .exposure import Exposure, predict_exposure, stationary_reference


def make_sequence(seed):
    rng = np.random.default_rng(87000+seed)
    sigma = float(np.exp(rng.uniform(np.log(.001), np.log(.004))))
    truth, camera, _, background = fixture(sigma, float(rng.uniform(-.01, .01)), bool(seed % 2))
    initial = {k: v.clone() for k, v in truth.items()}
    initial['spatial_means'] += initial['spatial_means'].new_tensor([.015, .02, .06])
    initial['temporal_scales'] += math.log(1.7)
    # Frame metadata lives on a fixed .002 grid. Choose nearby frames using
    # INITIAL model time centers, not true widths or errors. Time centers are
    # known/fixed in this fixture, as in the archived joint experiment.
    centers = sorted(set(round(float(t)/.002)*.002 for t in initial['temporal_means'][:, 0]))
    groups = []
    for shift in [-.002, 0., .002]:
        group = []
        for center in centers:
            shutter = Exposure(center+shift-.001, center+shift+.001)
            target = stationary_reference(truth, camera, background, shutter)
            group.append((shutter, target))
        groups.append(group)
    long_shutter = Exposure(.3, .7)
    long_target = stationary_reference(truth, camera, background, long_shutter)
    check_cameras = [camera_at(camera, x) for x in [.025, -.15, .15]]
    times = truth['temporal_means'][:, 0].unique().tolist()
    sharp_targets = [render_cuda(truth, cam, t, background).detach().double()
                     for cam in check_cameras for t in times]
    # Held-out time halfway between grid centers, at an unseen camera.
    check_shutters = [Exposure(c, c+.002) for c in centers]
    check_targets = [stationary_reference(truth, check_cameras[0], background, e) for e in check_shutters]
    return dict(seed=seed, sigma=sigma, truth=truth, initial=initial, camera=camera, background=background,
                groups=groups, long=[[(long_shutter, long_target)]], check_cameras=check_cameras,
                times=times, sharp_targets=sharp_targets, check_shutters=check_shutters,
                check_targets=check_targets)


@torch.no_grad()
def evaluate_sequence(raw, case):
    images = [render_cuda(raw, cam, t, case['background']).double()
              for cam in case['check_cameras'] for t in case['times']]
    errors = torch.stack([(a-b).square() for a, b in zip(images, case['sharp_targets'])])
    exposure_errors = [
        (stationary_reference(raw, case['check_cameras'][0], case['background'], e)-target).square()
        for e, target in zip(case['check_shutters'], case['check_targets'])]
    exposure_errors = torch.stack(exposure_errors)
    return dict(sharp_check_mse=float(errors.mean()), sharp_roi_check_mse=float(errors[:, :, 12:20, 12:20].mean()),
                exposure_check_mse=float(exposure_errors.mean()),
                **parameter_errors(raw, case['truth']))


def fit_sequence(case, arm, budget, steps_limit=None):
    groups = case['long'] if arm == 'long-partition32' else case['groups']
    method = 'gauss' if arm == 'sequence-gauss8' else 'partition'
    calls = 32 if arm == 'long-partition32' else 8
    sync(); start = time.perf_counter()
    state = JointParameters(case['initial'])
    optimizer = state.optimizer(lr=.05, spatial_lr=.01)
    steps = 0
    last_accepted = 0.
    last_loss = None
    rollback = False
    while steps_limit is None or steps < steps_limit:
        previous = state.snapshot()
        optimizer.zero_grad(set_to_none=True)
        group = groups[steps % len(groups)]
        loss = 0.
        # Accumulate the full group before one joint update so staggered
        # primitives are not starved by sparse frame-wise sampling.
        for shutter, target in group:
            image = predict_exposure(state.raw(), case['camera'], case['background'], shutter, calls, method)
            term = (image-target).square().mean()/len(group)
            if not bool(torch.isfinite(term)):
                raise ValueError('nonfinite observation loss')
            term.backward()
            loss += float(term.detach())
        optimizer.step(); sync()
        elapsed = time.perf_counter()-start
        if budget is not None and elapsed > budget:
            state.restore(previous)
            rollback = True
            break
        steps += 1
        last_accepted = elapsed
        last_loss = loss
    raw = {k: v.detach() for k, v in state.raw().items()}
    return dict(arm=arm, accepted_steps=steps, elapsed=elapsed, last_accepted_seconds=last_accepted,
                rollback=rollback,
                max_render_calls_per_step=calls*max(len(group) for group in groups),
                distinct_train_exposures=sum(len(group) for group in groups), last_loss=last_loss,
                final_spatial_means=raw['spatial_means'].tolist(),
                final_temporal_scales=raw['temporal_scales'].tolist(), **evaluate_sequence(raw, case))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--budget', type=float, default=6.)
    parser.add_argument('--steps', type=int, help='equal-update diagnostic; measured times need not match')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.budget <= 0 or (args.steps is not None and args.steps < 1):
        parser.error('need a new output path and a positive budget')
    torch.set_num_threads(4)
    from FasterGS4DCudaBackend import _C
    binary = Path(_C.__file__)
    sources = [Path(__file__), Path(__file__).with_name('exposure.py'), Path(__file__).with_name('joint_optimization.py')]
    output = dict(protocol='known-exposure information ablation; additional frames, not a same-data advantage',
                  budget_seconds=None if args.steps else args.budget, steps=args.steps,
                  initialization_xyz=[.015, .02, .06],
                  torch_version=torch.__version__, device=torch.cuda.get_device_name(0),
                  backend_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  spatial_lr=.01, temporal_lr=.05, short_duration=.002, frame_grid=.002,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.ones(1, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([dummy]); dummy.sum().backward(); optimizer.step(); sync()
    for seed in args.seeds:
        case = make_sequence(seed)
        record = dict(seed=seed, sigma=case['sigma'], initial=evaluate_sequence(case['initial'], case),
                      training_shutters=[[[e.lo, e.hi] for e, _ in group] for group in case['groups']], rows=[])
        output['cases'].append(record)
        arms = ['long-partition32', 'sequence-partition8', 'sequence-gauss8']
        offset = seed % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            row = fit_sequence(case, arm, None if args.steps else args.budget, args.steps)
            record['rows'].append(row)
            args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
            print(json.dumps(dict(seed=seed, **{k: v for k, v in row.items() if not k.startswith('final_')})), flush=True)


if __name__ == '__main__':
    main()
