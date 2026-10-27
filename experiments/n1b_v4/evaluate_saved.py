"""Evaluate a fixed checkpoint without training; supports reserved-camera audit."""
import argparse
import json
from pathlib import Path
import torch
from .run import Framework, FasterGS4DModel, FasterGS4DRenderer, ResidualGaussians, ResidualRenderer, DevelopmentDataset, evaluate, sha, ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--secondary', action='store_true')
    p.add_argument('--verify-primary', action='store_true')
    a = p.parse_args()
    out = Path(a.run).resolve()
    source = json.loads((out/'training.json').read_text())
    full = source['args'].get('full', False)
    assert not (full and a.secondary)
    assert not (a.secondary and a.verify_primary)
    destination = out/('reload_check.json' if a.verify_primary else ('reserved.json' if a.secondary else 'results.json'))
    assert not destination.exists(), destination
    Framework.setup(config_path=out/'config.yaml')
    torch.set_num_threads(4)
    state = torch.load(out/'final.pt', map_location='cpu', weights_only=True)
    assert sha(out/'final.pt') == source['checkpoint_sha256']
    model = FasterGS4DModel(out.name).build()
    model.gaussians = g = ResidualGaussians(model.SH_DEGREE, False)
    for key, value in state['raw'].items():
        setattr(g, '_'+key, torch.nn.Parameter(value.cuda()))
    g.active_sh_degree = state.get('active_sh_degree', 3)
    g.active_sh_bases = (g.active_sh_degree+1)**2
    g.residuals.update({key:torch.nn.Parameter(value.cuda()) for key,value in state['residuals'].items()})
    g.point_groups = [key for key in g.residuals if key != 'affine']
    if state['sigma_t'] is not None:
        g.reference_sigma_t = state['sigma_t'].cuda()
        g.reference_sigma_xyz = state['sigma_xyz'].cuda()
    # Also audit the native arm through the original inference entry point.
    renderer = ResidualRenderer(model) if len(g.residuals) else FasterGS4DRenderer(model)
    source['evaluation_renderer'] = type(renderer).__name__
    if full:
        from experiments.n1b_native.data import install_coffee_cache
        from Datasets.PlenopticVideoBlender import CustomDataset
        install_coffee_cache(ROOT)
        dataset = CustomDataset(Framework.config.DATASET.PATH)
        views = dataset.test()
    else:
        dataset = DevelopmentDataset(Framework.config.DATASET.PATH)
        dataset.preload(secondary=a.secondary)
        views = dataset.secondary if a.secondary else dataset.data['val']
    source['evaluation_split'] = 'official_test' if full else ('reserved' if a.secondary else 'primary')
    source['evaluation_only'] = True
    source['validation'] = evaluate(renderer, views)
    if a.verify_primary:
        expected = json.loads((out/'results.json').read_text())['validation']['mean']
        delta = {k:source['validation']['mean'][k]-expected[k] for k in expected}
        assert abs(delta['psnr']) < .001 and abs(delta['ssim']) < 1e-5 and abs(delta['lpips']) < 1e-5, delta
        source['reload_mean_delta'] = delta
        if full:
            from Datasets.utils import save_image
            for index in (0,150,299):
                view = views[index]
                save_image(out/f'audit_view{index}_prediction.png',
                           renderer.render_image_inference(view, True)['rgb'])
                if not len(g.residuals):
                    save_image(out/f'audit_view{index}_target.png', view.rgb)
    destination.write_text(json.dumps(source, indent=2))
    print(json.dumps({'file':str(destination),'validation':source['validation']['mean']}), flush=True)

if __name__ == '__main__':
    main()
