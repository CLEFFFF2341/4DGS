"""Joint-fit causal ablations with preserved results and independent evaluation.

Uses the archived synthetic fixture and native renderer. No real checkpoint is
loaded. Evaluation/ground truth never chooses an update, learning rate or stop.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.n1b.narrow_benchmark import fixture, event_nodes
from experiments.n1b.benchmark import sync
from .joint_optimization import JointParameters
from .train_probe import prediction, render_pixels


def camera_at(cam, x):
    W = cam.w2c.clone()
    W[0, 3] -= x
    return replace(cam, w2c=W, position=cam.position + cam.position.new_tensor([x, 0, 0]))


@torch.no_grad()
def reference_image(raw, cam, pix, bg, order=8):
    image = torch.zeros((len(pix), 3), device=bg.device, dtype=torch.double)
    for t, w in event_nodes(raw, cam, pix, bg, order):
        image.add_(render_pixels(raw, cam, pix, bg, t), alpha=w)
    return image


def make_case(seed, views=1, initialization='x-only', full_frame=False):
    rng = np.random.default_rng(87000 + seed)
    sigma = float(np.exp(rng.uniform(np.log(.001), np.log(.004))))
    truth, cam, pix, bg = fixture(sigma, float(rng.uniform(-.01, .01)), bool(seed % 2))
    if full_frame:
        yy, xx = torch.meshgrid(torch.arange(cam.height, device=bg.device),
                               torch.arange(cam.width, device=bg.device), indexing='ij')
        pix = torch.stack((xx.flatten() + .5, yy.flatten() + .5), -1)
    initial = {k: v.clone() for k, v in truth.items()}
    initial['temporal_scales'] += math.log(1.7)
    initial['spatial_means'][:, 0] += .015
    if initialization == 'xyz':
        # Stress test: correct initial depth must not be assumed by a repair.
        initial['spatial_means'][:, 1] += .02
        initial['spatial_means'][:, 2] += .06
    cameras = [cam] if views == 1 else [cam, camera_at(cam, -.03)]
    checks = [camera_at(cam, .025), camera_at(cam, -.15), camera_at(cam, .15)]
    targets = [reference_image(truth, c, pix, bg) for c in cameras]
    check_targets = [reference_image(truth, c, pix, bg) for c in checks]
    return dict(seed=seed, sigma=sigma, truth=truth, initial=initial, pix=pix, bg=bg,
                cameras=cameras, checks=checks, targets=targets, check_targets=check_targets)


@torch.no_grad()
def parameter_errors(raw, truth):
    xyz = (raw['spatial_means'] - truth['spatial_means']).square().mean(0)
    return dict(position_mse=float(xyz.mean()), xyz_mse=xyz.tolist(),
                log_scale_mse=float((raw['temporal_scales'] - truth['temporal_scales']).square().mean()))


@torch.no_grad()
def evaluate(raw, case):
    pix = case['pix']
    roi = ((pix >= 12) & (pix < 20)).all(-1)
    train_errors = [(reference_image(raw, c, pix, case['bg']) - target).square()
                    for c, target in zip(case['cameras'], case['targets'])]
    check_errors = [(reference_image(raw, c, pix, case['bg']) - target).square()
                    for c, target in zip(case['checks'], case['check_targets'])]
    train = [float(e.mean()) for e in train_errors]
    check = [float(e.mean()) for e in check_errors]
    return dict(reference_train_mse=train, reference_check_mse=check,
                roi_train_mse=[float(e[roi].mean()) for e in train_errors],
                roi_check_mse=[float(e[roi].mean()) for e in check_errors],
                **parameter_errors(raw, case['truth']))


def fit(case, method, calls, mode, budget, steps_limit, spatial_lr=None, temporal_lr=.01, warmup=0):
    torch.manual_seed(90000 + case['seed'])
    sync()
    start = time.perf_counter()
    params = JointParameters(case['initial'], mode)
    optimizer = params.optimizer(lr=temporal_lr, spatial_lr=spatial_lr)
    accepted = 0
    trajectory = []
    last_loss = None
    last_accepted_seconds = 0.
    rollback = False
    while steps_limit is None or accepted < steps_limit:
        previous = params.snapshot()
        optimizer.zero_grad(set_to_none=True)
        raw = params.raw()
        view = accepted % len(case['cameras'])
        predicted = prediction(method, raw, case['cameras'][view], case['pix'], case['bg'], calls)
        loss = (predicted - case['targets'][view]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite joint loss')
        loss.backward()
        if accepted < warmup:
            params.position.grad = None
        optimizer.step()
        sync()
        elapsed = time.perf_counter() - start
        if budget is not None and elapsed > budget:
            params.restore(previous)
            rollback = True
            break
        accepted += 1
        last_loss = float(loss.detach())
        last_accepted_seconds = elapsed
        if accepted in (1, 5, 20, 50, 100, 200):
            trajectory.append(dict(step=accepted, loss_before_update=last_loss,
                                   **parameter_errors(params.raw(), case['truth'])))
    sync()
    elapsed = time.perf_counter() - start
    raw = {k: v.detach() for k, v in params.raw().items()}
    return dict(method=method, calls=calls, coordinates=mode, spatial_lr=spatial_lr,
                temporal_lr=temporal_lr, temporal_warmup_steps=warmup,
                accepted_steps=accepted, elapsed=elapsed, last_accepted_seconds=last_accepted_seconds,
                rollback=rollback, last_estimated_loss=last_loss, trajectory=trajectory,
                final_spatial_means=raw['spatial_means'].tolist(),
                final_temporal_scales=raw['temporal_scales'].tolist(), **evaluate(raw, case))


def diagnose(case):
    raw = {k: v.clone().requires_grad_() for k, v in case['initial'].items()}
    cam, target, pix, bg = case['cameras'][0], case['targets'][0], case['pix'], case['bg']
    image = reference_image(raw, cam, pix, bg)
    residual_weight = 2 * (image - target) / image.numel()
    grads = [torch.zeros_like(raw[k]) for k in ('spatial_means', 'temporal_scales')]
    for t, w in event_nodes(raw, cam, pix, bg, 8):
        value = (render_pixels(raw, cam, pix, bg, t) * residual_weight).sum() * w
        gs = torch.autograd.grad(value, (raw['spatial_means'], raw['temporal_scales']))
        for g, add in zip(grads, gs):
            g.add_(add)
    # Compare same physical perturbations, recomputing independent event cuts.
    derivatives = []
    for key, axis, step in [('spatial_means', 0, 2e-4), ('spatial_means', 2, 2e-3),
                            ('temporal_scales', 0, 2e-3)]:
        values = []
        for sign in [-1, 1]:
            trial = {k: v.detach().clone() for k, v in raw.items()}
            trial[key][:, axis] += sign * step
            values.append(float((reference_image(trial, cam, pix, bg) - target).square().mean()))
        g = grads[0 if key == 'spatial_means' else 1][:, axis].sum()
        derivatives.append(dict(parameter=key, axis=axis, step=step, native_branch_ad=float(g),
                                finite_difference=(values[1] - values[0]) / (2 * step)))
    # Local image Jacobian of the *integrated* objective, in scale-aware units.
    coordinates = JointParameters(case['initial'])
    params = coordinates.parameters()
    columns = []
    h = .01
    with torch.no_grad():
        for parameter in params:
            for i in range(parameter.numel()):
                original = parameter.flatten()[i].clone()
                parameter.flatten()[i] = original - h
                minus = reference_image(coordinates.raw(), cam, pix, bg)
                parameter.flatten()[i] = original + h
                plus = reference_image(coordinates.raw(), cam, pix, bg)
                parameter.flatten()[i] = original
                columns.append(((plus - minus) / (2 * h)).flatten())
    J = torch.stack(columns, 1)
    singular = torch.linalg.svdvals(J)
    geom, temporal = J[:, :24], J[:, 24:]
    z = geom[:, 2::3]
    projected = temporal @ torch.linalg.lstsq(temporal, z).solution
    return dict(initial=evaluate(case['initial'], case),
                reference_order4_8_mse=float((reference_image(raw, cam, pix, bg, 4) - image).square().mean()),
                native_gradient_xyz_rms=grads[0].square().mean(0).sqrt().tolist(),
                native_gradient_log_scale_rms=float(grads[1].square().mean().sqrt()),
                integrated_loss_derivatives=derivatives,
                scaled_jacobian_singular_values=singular.tolist(),
                scaled_jacobian_condition=float(singular[0] / singular[-1]),
                depth_jacobian_fraction_explained_by_temporal=float(projected.square().sum() / z.square().sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=float, default=3.)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--views', type=int, choices=[1, 2], default=1)
    parser.add_argument('--initialization', choices=['x-only', 'xyz'], default='x-only')
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--full-frame', action='store_true')
    parser.add_argument('--temporal-lr', type=float, default=.01)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--arms', nargs='+', default=['partition:32:legacy', 'partition:32:scale-aware',
                                                   'gauss:128:legacy', 'gauss:128:scale-aware'])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new path to preserve results')
    if args.budget <= 0 or (args.steps is not None and args.steps < 1) or args.repeats < 1:
        parser.error('positive budget/steps required')
    parsed_arms = []
    for arm in args.arms:
        fields = arm.split(':')
        if len(fields) not in (3, 4, 5) or fields[0] not in ('partition', 'gauss') or fields[2] not in ('legacy', 'scale-aware'):
            parser.error('arm must be partition|gauss:calls:legacy|scale-aware[:spatial_lr[:temporal_lr]]')
        calls = int(fields[1])
        spatial_lr = float(fields[3]) if len(fields) >= 4 else None
        temporal_lr = float(fields[4]) if len(fields) == 5 else args.temporal_lr
        if calls < 1 or temporal_lr <= 0 or (spatial_lr is not None and spatial_lr <= 0):
            parser.error('positive calls and learning rates required')
        parsed_arms.append((fields[0], calls, fields[2], spatial_lr, temporal_lr))
    torch.set_num_threads(4)
    from FasterGS4DCudaBackend import _C
    binary = Path(_C.__file__)
    sources = [Path(__file__), Path(__file__).with_name('joint_optimization.py'),
               Path(__file__).with_name('train_probe.py'), Path(__file__).with_name('partition.py')]
    out = dict(protocol='joint audit v2; controls receive identical updates; evaluation never controls training',
               budget_seconds=None if args.steps else args.budget, steps=args.steps, views=args.views,
               initialization=args.initialization, full_frame=args.full_frame, torch_version=torch.__version__,
               repeats=args.repeats, arms=args.arms,
               source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               backend_path=str(binary), backend_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
               check_camera_x=[.025, -.15, .15], cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.ones(1, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([dummy]); dummy.sum().backward(); opt.step(); sync()
    for seed in args.seeds:
        case = make_case(seed, args.views, args.initialization, args.full_frame)
        result = dict(seed=seed, sigma=case['sigma'], initial=evaluate(case['initial'], case), rows=[])
        if args.diagnose:
            result['diagnostics'] = diagnose(case)
            print(json.dumps(dict(seed=seed, diagnostics=result['diagnostics'])), flush=True)
        out['cases'].append(result)
        for repeat in range(args.repeats):
            # Rotate order to avoid always assigning the last/hottest slot to one arm.
            offset = (seed + repeat) % len(parsed_arms)
            arms = parsed_arms[offset:] + parsed_arms[:offset]
            for method, calls, mode, spatial_lr, temporal_lr in arms:
                row = fit(case, method, calls, mode, None if args.steps else args.budget, args.steps,
                          spatial_lr, temporal_lr, args.warmup)
                row['repeat'] = repeat
                result['rows'].append(row)
                args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
                print(json.dumps(dict(seed=seed, **{k: v for k, v in row.items() if k not in
                                 ('trajectory', 'final_spatial_means', 'final_temporal_scales')})), flush=True)


if __name__ == '__main__':
    main()
