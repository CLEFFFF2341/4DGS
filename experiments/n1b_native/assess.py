"""Pair only completed full runs; retain per-image records and integrity checks."""
import json
from pathlib import Path
import shutil
import numpy as np
from .run import sha


def main():
    root = Path('output/n1b_native')
    dest = Path('experiments/n1b_native/results')
    dest.mkdir(exist_ok=True)
    pairs = []
    for path in sorted(root.glob('*_native/results.json')):
        candidate = path.parent.with_name(path.parent.name.removesuffix('_native') + '_field')
        if not (candidate / 'results.json').exists():
            continue
        records = [json.loads(p.read_text()) for p in (path, candidate / 'results.json')]
        if any(r['steps'] != 30000 or 'metrics' not in r for r in records):
            continue
        n, c = records
        row = {'name': path.parent.name.removesuffix('_native'), 'dataset': n['args']['dataset'],
               'seed': n['args']['seed'], 'initial_hash_match': n['initial_sha256'] == c['initial_sha256'],
               'arms': {}, 'delta': {}}
        assert row['initial_hash_match']
        for label, folder, record in [('native', path.parent, n), ('field', candidate, c)]:
            assert sha(folder / 'final.pt') == record['checkpoint_sha256']
            assert record['callback_counts']['training_iteration'] == 30000
            assert record['callback_counts']['densify'] == 144
            assert record['callback_counts']['morton_ordering'] == 4
            row['arms'][label] = {'metrics': record['metrics']['mean'],
                'gaussians': record['final_count'],
                'training_seconds': sum(v for k, v in record['callback_seconds'].items()
                    if k in ('training_iteration', 'densify', 'morton_ordering', 'reset_opacities',
                             'reset_opacities_extra', 'increase_sh_degree', 'step_temporal_field')),
                'checkpoint_sha256': record['checkpoint_sha256']}
            shutil.copy2(folder / 'results.json', dest / (folder.name + '.json'))
        for rendering in n['metrics']['mean']:
            row['delta'][rendering] = {precision: {metric:
                c['metrics']['mean'][rendering][precision][metric] - n['metrics']['mean'][rendering][precision][metric]
                for metric in ('psnr', 'ssim', 'lpips')} for precision in ('float', '8bit')}
        pairs.append(row)
    summary = {}
    for dataset in sorted(set(p['dataset'] for p in pairs)):
        selected = [p for p in pairs if p['dataset'] == dataset]
        summary[dataset] = {'pairs': len(selected),
            'positive_psnr_pairs': sum(p['delta']['midpoint']['8bit']['psnr'] > 0 for p in selected),
            'mean_midpoint_8bit_delta': {metric: float(np.mean([
                p['delta']['midpoint']['8bit'][metric] for p in selected])) for metric in ('psnr', 'ssim', 'lpips')}}
    report = {'summary': summary, 'pairs': pairs}
    (dest / 'assessment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
