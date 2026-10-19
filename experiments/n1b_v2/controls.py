"""Original non-narrow counterexamples: identical-call regression and timing."""
import argparse
import json
from pathlib import Path
import torch
from experiments.n1b.scenes import scene
from experiments.n1b.narrow_benchmark import timed
from .pilot import measure


def main(name):
    torch.set_num_threads(4)
    raw, cam, pix, bg = scene(name)
    raw = {k: v.requires_grad_() for k, v in raw.items()}
    gen = torch.Generator(device='cuda').manual_seed(339)
    weight = torch.randn(64, 3, generator=gen, device='cuda', dtype=torch.double)/192
    result = {'scene': name, 'rows': [], 'scope': 'same-call regression, not equal-time superiority'}
    for gradient in [False, True]:
        reference, _ = timed('gauss', 2048, raw, cam, pix, bg, weight, gradient)
        lower, _ = timed('gauss', 1024, raw, cam, pix, bg, weight, gradient)
        for calls in [16, 64]:
            baseline, bt = timed('gauss', calls, raw, cam, pix, bg, weight, gradient)
            ours, ot, detail = measure(raw, cam, bg, weight, calls, gradient, 'balanced')
            result['rows'].append(dict(gradient=gradient, calls=calls,
                reference_change=float((reference-lower).square().mean()),
                gauss_mse=float((baseline-reference).square().mean()),
                partition_mse=float((ours-reference).square().mean()),
                direct_difference=float((ours-baseline).abs().max()),
                gauss_seconds=bt, partition_seconds=ot, intervals=detail['intervals']))
    path = Path('experiments/n1b_v2/results')/f'control_{name}.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['single', 'double', 'triple', 'dense', 'static', 'static_layers', 'perspective', 'overflow'])
    main(parser.parse_args().name)
