"""One case per invocation keeps interrupted long tasks resumable."""
import argparse
import json
import time
from pathlib import Path
import torch
from experiments.n1b.narrow_benchmark import fixture, reference, timed, calibrate
from experiments.n1b.benchmark import sync
from .partition import estimate


def measure(raw, cam, bg, weight, k, gradient, allocation='equal'):
    sync(); start = time.perf_counter()
    with torch.set_grad_enabled(gradient):
        val, detail = estimate(raw, cam, bg, weight, k, gradient, allocation)
    sync()
    return val, time.perf_counter()-start, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=.001)
    parser.add_argument('--phase', type=float, default=.003)
    parser.add_argument('--staggered', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw, cam, pix, bg = fixture(args.sigma, args.phase, args.staggered)
    raw = {k: v.requires_grad_() for k, v in raw.items()}
    gen = torch.Generator(device='cuda').manual_seed(671)
    weight = torch.randn(64, 3, generator=gen, device='cuda', dtype=torch.double)/192
    r4, g4, n4 = reference(raw, cam, pix, bg, weight, 4)
    ref, gr, n8 = reference(raw, cam, pix, bg, weight, 8)
    report = {'fixture': vars(args), 'reference_nodes': [n4, n8],
              'reference_change': {'image': float((ref-r4).square().mean()), 'gradient': float((gr-g4).square().mean())}, 'rows': []}
    for gradient in [False, True]:
        for method in ['gauss', 'partition']:
            if method == 'gauss':
                k = calibrate('gauss', .05, raw, cam, pix, bg, weight, gradient)
            else:
                k = 16
                for _ in range(3):
                    _, dt, _ = measure(raw, cam, bg, weight, k, gradient)
                    k = max(2, min(256, int(k*.04/dt)))
            trials = []
            for repeat in range(3):
                if method == 'gauss':
                    value, dt = timed('gauss', k, raw, cam, pix, bg, weight, gradient)
                    detail = {}
                else:
                    value, dt, detail = measure(raw, cam, bg, weight, k, gradient)
                target = gr if gradient else ref
                trials.append({'mse': float((value-target).square().mean()), 'max_abs': float((value-target).abs().max()), 'seconds': dt, 'within_budget': dt <= .05})
            report['rows'].append({'gradient': gradient, 'method': method, 'calls': k, **detail, 'trials': trials})
    directory = Path('experiments/n1b_v2/results'); directory.mkdir(exist_ok=True)
    name = f'pilot_{args.sigma}_{args.phase}_{args.staggered}.json'
    (directory/name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
