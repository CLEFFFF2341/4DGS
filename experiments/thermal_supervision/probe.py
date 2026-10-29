"""Small controlled transactions on real CUDA renders, not a data benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .core import GateConfig
from .engine import SearchConfig, TransferEngine
from .profiled import ProfileConfig, ProfiledTransferEngine
from .synthetic import make_case


@torch.no_grad()
def metrics(model, truth, test):
    prediction = model.render(test)
    error = float((prediction-test.target).square().mean())
    center_error = float((model.raw['spatial_means']+model.thermal_delta-truth.raw['spatial_means']).square().mean().sqrt())
    return {'rgb_mse': error, 'rgb_psnr': -10*__import__('math').log10(max(error, 1e-15)),
            'center_rmse': center_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--engine', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--cases', nargs='+', default=['geometry', 'appearance', 'mixed', 'unchanged'])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = (ProfileConfig if args.engine == 'v2' else SearchConfig)(gate=GateConfig(min_source_gain=1e-5, min_witness_gain=1e-5,
        appearance_advantage_margin=1e-5, max_source_change=2., max_rgb_loss_increase=1e-6,
        max_rgb_anchor_increase=1e-6))
    report = {'scope': 'synthetic known 20-Gaussian scenes, same native renderer for generation/evaluation; no real-data claim',
              'config': asdict(config), 'steps': args.steps, 'rows': [],
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    for seed in args.seeds:
        for case in args.cases:
            truth, model, source, witness, guard, test = make_case(seed, case)
            engine = (ProfiledTransferEngine if args.engine == 'v2' else TransferEngine)(model, config)
            initial = metrics(model, truth, test)
            torch.cuda.synchronize()
            started = time.perf_counter()
            transactions = [engine.step(source, [witness], [guard]) for _ in range(args.steps)]
            torch.cuda.synchronize()
            row = {'seed': seed, 'case': case, 'before': initial, 'after': metrics(model, truth, test),
                   'seconds': time.perf_counter()-started, 'transactions': transactions}
            report['rows'].append(row)
            (out/'results.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({'seed': seed, 'case': case, 'accepted': sum(t['accepted'] for t in transactions),
                              'before': initial, 'after': row['after'], 'reasons': [t['reason'] for t in transactions]}), flush=True)


if __name__ == '__main__':
    main()
