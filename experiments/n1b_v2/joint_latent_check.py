"""Post-hoc instantaneous-image diagnostic; never used to select training steps.

Samples each true pulse peak at unseen cameras. These sharp images are not
training targets. This adds a diagnostic beyond the frozen confirmation protocol.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiments.n1b.core import render_cuda
from experiments.n1b.narrow_benchmark import fixture
from .joint_probe import camera_at


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    directory = Path(__file__).parent / 'results' / 'joint_audit'
    destination = directory / 'instantaneous_check.json'
    if destination.exists():
        raise FileExistsError(destination)
    data = json.loads((directory / 'confirmation.json').read_text(encoding='utf-8'))
    output = []
    for saved in data['cases']:
        seed = saved['seed']
        rng = np.random.default_rng(87000 + seed)
        sigma = float(np.exp(rng.uniform(np.log(.001), np.log(.004))))
        truth, cam, pix, bg = fixture(sigma, float(rng.uniform(-.01, .01)), bool(seed % 2))
        initial = {k: v.clone() for k, v in truth.items()}
        initial['spatial_means'][:, 0] += .015
        initial['temporal_scales'] += math.log(1.7)
        times = truth['temporal_means'][:, 0].unique().tolist()
        cameras = [camera_at(cam, x) for x in [.025, -.15, .15]]
        targets = [render_cuda(truth, camera, t, bg) for camera in cameras for t in times]
        models = [('initial', initial)]
        for row in saved['rows']:
            raw = {k: v.clone() for k, v in initial.items()}
            raw['spatial_means'] = raw['spatial_means'].new_tensor(row['final_spatial_means'])
            raw['temporal_scales'] = raw['temporal_scales'].new_tensor(row['final_temporal_scales'])
            name = f"{row['method']}{row['calls']}:{row['coordinates']}:{row['temporal_lr']}"
            models.append((name, raw))
        rows = []
        for name, raw in models:
            images = [render_cuda(raw, camera, t, bg) for camera in cameras for t in times]
            errors = torch.stack([(a.double()-b.double()).square() for a, b in zip(images, targets)])
            rows.append(dict(model=name, full_mse=float(errors.mean()),
                             roi_mse=float(errors[:, :, 12:20, 12:20].mean())))
        result = dict(seed=seed, times=times, rows=rows)
        output.append(result)
        print(json.dumps(result), flush=True)
    destination.write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
