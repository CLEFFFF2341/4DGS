"""Audit real training MSE gradients against dense native branch AD quadrature."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.n1b_v2.real_dynamic import DynamicData, predict


def estimate(initial, data, side, index, calls):
    raw = {k: v.detach().clone().requires_grad_() for k, v in initial.items()}
    torch.cuda.synchronize()
    start = time.perf_counter()
    image, _ = predict(raw, data, side, index, calls, 'gauss')
    loss = (image - data.images[side][index]).square().mean()
    gradient = torch.autograd.grad(loss, tuple(raw.values()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return image.detach(), {k: g.detach() for k, g in zip(raw, gradient)}, float(loss), elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', default=['output/dsec_c120_s0', 'output/dsec_n120_s0'])
    p.add_argument('--states', nargs='+', default=['shared_initial', 'gauss'])
    p.add_argument('--output', required=True)
    a = p.parse_args()
    path = Path(a.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {'semantics': 'MSE training-loss gradients of native branch AD; no moving-boundary derivative',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows': []}
    for run in a.runs:
        previous = json.loads((Path(run) / 'results.json').read_text())['args']
        data = DynamicData(previous['root'], previous['start'], previous['count'], previous['width'])
        # Fixed training observations, both cameras, spread through each clip.
        observations = [data.train[j] for j in (0, 11, 24, 35)]
        for state in a.states:
            model_path = Path(run) / (state + '.pt')
            initial = torch.load(model_path, weights_only=True)
            for side, index in observations:
                fine, ref, loss, sec = estimate(initial, data, side, index, 64)
                row = {'run': run, 'state': state, 'side': side, 'frame': data.frames[index],
                       'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
                       'reference_loss': loss, 'reference_seconds': sec, 'orders': {}}
                for calls in (1, 2, 4, 8, 16, 32):
                    image, grads, objective, elapsed = estimate(initial, data, side, index, calls)
                    groups = {}
                    for key in ref:
                        r = ref[key].double().flatten(); g = grads[key].double().flatten()
                        norm = r.square().sum()
                        groups[key] = {'relative_l2': float((g-r).norm()/r.norm().clamp_min(1e-30)),
                                       'cosine': float(torch.dot(g, r)/(g.norm()*r.norm()).clamp_min(1e-30)),
                                       'reference_norm': float(norm.sqrt()),
                                       'absolute_mse': float((g-r).square().mean())}
                    row['orders'][str(calls)] = {'image_mse': float((image-fine).square().mean()),
                                                'loss': objective, 'seconds': elapsed, 'groups': groups}
                result['rows'].append(row)
                path.write_text(json.dumps(result, indent=2))
                print(json.dumps({'run': run, 'state': state, 'frame': data.frames[index], 'side': side,
                                  'mse8': row['orders']['8']['image_mse'],
                                  'max_gradient_rel8': max(g['relative_l2'] for g in row['orders']['8']['groups'].values())}), flush=True)
    summary = {}
    for calls in (1, 2, 4, 8, 16, 32):
        rows = [r['orders'][str(calls)] for r in result['rows']]
        summary[str(calls)] = {'max_image_mse': max(r['image_mse'] for r in rows),
                               'group_relative_l2_median': float(np.median([g['relative_l2'] for r in rows for g in r['groups'].values()])),
                               'group_relative_l2_max': max(g['relative_l2'] for r in rows for g in r['groups'].values())}
    result['summary'] = summary
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
