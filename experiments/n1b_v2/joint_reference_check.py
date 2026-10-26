"""Independent order8/16 native evaluation check on saved fitted parameters."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.n1b.narrow_benchmark import fixture
from .joint_probe import camera_at
from .exposure import Exposure, stationary_reference


def main():
    torch.set_num_threads(4)
    directory = Path(__file__).parent / 'results' / 'joint_audit'
    destination = directory / 'reference_convergence.json'
    if destination.exists():
        raise FileExistsError(destination)
    confirmation = json.loads((directory / 'confirmation.json').read_text(encoding='utf-8'))
    results = []
    for saved in confirmation['cases']:
        if saved['seed'] not in (6, 7):
            continue
        rng = np.random.default_rng(87000+saved['seed'])
        sigma = float(np.exp(rng.uniform(np.log(.001), np.log(.004))))
        truth, camera, pixels, background = fixture(sigma, float(rng.uniform(-.01, .01)), bool(saved['seed'] % 2))
        models = [('truth', truth)]
        for row in saved['rows']:
            if row['coordinates'] != 'scale-aware' or row['temporal_lr'] != .05:
                continue
            raw = {k: v.clone() for k, v in truth.items()}
            raw['spatial_means'] = raw['spatial_means'].new_tensor(row['final_spatial_means'])
            raw['temporal_scales'] = raw['temporal_scales'].new_tensor(row['final_temporal_scales'])
            models.append((f"{row['method']}{row['calls']}", raw))
        for name, raw in models:
            # Main train camera and the original unseen check camera.
            for camera_index, cam in enumerate([camera, camera_at(camera, .025)]):
                low = stationary_reference(raw, cam, background, Exposure(.3, .7), 8)
                high = stationary_reference(raw, cam, background, Exposure(.3, .7), 16)
                row = dict(seed=saved['seed'], model=name, camera=camera_index,
                           merged_identical_float32_timestamps=True,
                           full_mse_change=float((low-high).square().mean()),
                           roi_mse_change=float((low[:, 12:20, 12:20]-high[:, 12:20, 12:20]).square().mean()),
                           max_abs_change=float((low-high).abs().max()))
                results.append(row)
                destination.write_text(json.dumps(results, indent=2), encoding='utf-8')
                print(json.dumps(row), flush=True)


def sequence_check():
    torch.set_num_threads(4)
    directory = Path(__file__).parent / 'results' / 'joint_audit'
    destination = directory / 'sequence_reference_convergence.json'
    if destination.exists():
        raise FileExistsError(destination)
    data = json.loads((directory / 'sequence_confirmation.json').read_text(encoding='utf-8'))
    results = []
    for saved in data['cases']:
        if saved['seed'] not in (14, 15):
            continue
        rng = np.random.default_rng(87000+saved['seed'])
        sigma = float(np.exp(rng.uniform(np.log(.001), np.log(.004))))
        truth, camera, _, background = fixture(sigma, float(rng.uniform(-.01, .01)), bool(saved['seed'] % 2))
        models = [('truth', truth)]
        for row in saved['rows']:
            if row['arm'] == 'long-partition32':
                continue
            raw = {k: v.clone() for k, v in truth.items()}
            raw['spatial_means'] = raw['spatial_means'].new_tensor(row['final_spatial_means'])
            raw['temporal_scales'] = raw['temporal_scales'].new_tensor(row['final_temporal_scales'])
            models.append((row['arm'], raw))
        for name, raw in models:
            for index, (lo, hi) in enumerate(saved['training_shutters'][1]):
                exposure = Exposure(lo+.001, hi+.001)
                low = stationary_reference(raw, camera_at(camera, .025), background, exposure, 8)
                high = stationary_reference(raw, camera_at(camera, .025), background, exposure, 16)
                row = dict(seed=saved['seed'], model=name, shutter=index,
                           full_mse_change=float((low-high).square().mean()),
                           max_abs_change=float((low-high).abs().max()))
                results.append(row)
    destination.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(dict(comparisons=len(results), max_mse_change=max(r['full_mse_change'] for r in results),
                         max_abs_change=max(r['max_abs_change'] for r in results))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', action='store_true')
    args = parser.parse_args()
    sequence_check() if args.sequence else main()
