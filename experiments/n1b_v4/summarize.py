"""Compact, auditable metrics; never equate development gains with confirmation."""
import json
from pathlib import Path

ROOT = Path('output/n1b_v4')
DEST = Path('experiments/n1b_v4/results')

def main():
    rows = []
    for path in sorted([*ROOT.glob('*/results.json'), *ROOT.glob('*/reserved.json')]):
        r = json.loads(path.read_text())
        if 'validation' not in r:
            continue
        seed = r['args']['seed']
        full = r['args'].get('full', False)
        base_path = ROOT/f'{"full" if full else "dev"}_s{seed}_native'/path.name
        if not base_path.exists():
            continue
        b = json.loads(base_path.read_text())
        assert r['prefix_sha256'] == b['prefix_sha256']
        assert r['shared_prefix']['initial_tensor_sha256'] == b['shared_prefix']['initial_tensor_sha256']
        if not full:
            assert r['pixel_hash'] == b['pixel_hash']
        metrics = r['validation']['mean']
        assert [v['image'] for v in r['validation']['views']] == [v['image'] for v in b['validation']['views']]
        delta = {k:metrics[k]-b['validation']['mean'][k] for k in metrics}
        label = path.parent.name + ('@reserved' if path.name == 'reserved.json' else '')
        rows.append({'run':label, 'args':r['args'],
            'split':r.get('evaluation_split','primary'),
            'mean':metrics,'delta_vs_native':delta,
            'passes_primary_gate':delta['psnr'] >= .1 and delta['ssim'] >= 0 and delta['lpips'] <= 0,
            'positive_psnr_views':sum(x['psnr'] > y['psnr'] for x,y in zip(r['validation']['views'],b['validation']['views'])),
            'num_views':len(r['validation']['views']),
            'extra_parameters':r['shared_prefix']['extra_parameters'],
            'gaussians':r['final_gaussians'], 'continuation_seconds':r['seconds'],
            'source_sha256':r['source_sha256'], 'prefix_sha256':r['prefix_sha256'],
            'checkpoint_sha256':r['checkpoint_sha256'], 'views':r['validation']['views']})
        print(f'{label:38s} {metrics["psnr"]:.6f}  delta {delta["psnr"]:+.6f} / {delta["ssim"]:+.6f} / {delta["lpips"]:+.6f}')
    DEST.mkdir(exist_ok=True)
    (DEST/'development.json').write_text(json.dumps(rows,indent=2))

if __name__ == '__main__':
    main()
