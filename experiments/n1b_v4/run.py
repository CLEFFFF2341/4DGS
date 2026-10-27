"""Small real-data development using a held-out training camera."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
import yaml
import Framework
from Methods.FasterGS4D.Model import FasterGS4DModel
from Methods.FasterGS4D.Renderer import FasterGS4DRenderer
from experiments.n1b_native.run import sha
from experiments.n1b_native.native import raw_state
from .data import DevelopmentDataset
from .model import ResidualGaussians, ResidualRenderer
from .train import Trainer, resume, MSEAugmentedLoss


@torch.no_grad()
def evaluate(renderer, views):
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda().eval()
    renderer.model.eval()
    rows = []
    for view in views:
        a = renderer.render_image_inference(view, True)['rgb']
        b = view.rgb.cuda()
        a, b = [(v.clamp(0,1)*255+.5).byte().float()/255 for v in (a,b)]
        rows.append({'image': view._rgb.path.stem,
            'psnr': float(-10*torch.log10((a-b).square().mean())),
            'ssim': float(structural_similarity_index_measure(a[None], b[None], data_range=1.)),
            'lpips': float(lp(a[None], b[None]))})
        lp.reset()
    return {'mean': {k: float(np.mean([r[k] for r in rows])) for k in ('psnr','ssim','lpips')}, 'views': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prefix', action='store_true')
    p.add_argument('--checkpoint')
    p.add_argument('--kind', choices=['native','affine','accel','color','accel_color'], default='native')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--accel-lr', type=float, default=.01)
    p.add_argument('--color-lr', type=float, default=.001)
    p.add_argument('--mse-weight', type=float, default=0.)
    p.add_argument('--secondary', action='store_true')
    p.add_argument('--full', action='store_true', help='Frozen confirmation on all native training/test images')
    a = p.parse_args()
    assert a.prefix != bool(a.checkpoint)
    assert a.mse_weight >= 0 and (not a.prefix or a.mse_weight == 0)
    assert not (a.full and a.secondary)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT/'configs/fastergs4d_coffee_martini.yaml').read_text())
    config['GLOBAL'].update(RANDOM_SEED=a.seed, LOG_LEVEL=0)
    config['TRAINING'].update(NUM_ITERATIONS=a.steps, BATCH_SIZE=4 if a.full else 1, MODEL_NAME=out.name)
    config['TRAINING']['BACKUP'].update(FINAL_CHECKPOINT=False, TRAINING_STATE=False)
    config['DATASET']['IMAGE_SCALE_FACTOR'] = .5 if a.full else .125
    config_path = out/'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    sources = {str(path.relative_to(ROOT)): sha(path) for path in Path(__file__).parent.glob('*.py')}
    record = {'args': vars(a), 'source_sha256': sources,
              'commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()}
    record['native_source_sha256'] = {name:sha(ROOT/'src/Methods/FasterGS4D'/name)
                                     for name in ('Model.py','Renderer.py','Trainer.py','Loss.py')}
    (out/'protocol.json').write_text(json.dumps(record, indent=2))
    Framework.setup(config_path=config_path)
    torch.set_num_threads(4)
    # Keep the original allocator probe before any dataset/extra CUDA allocations.
    model = FasterGS4DModel(out.name).build()
    model.gaussians = ResidualGaussians(model.SH_DEGREE, False)
    model.output_directory = out
    renderer = ResidualRenderer(model)
    trainer = Trainer(model=model, renderer=renderer, prefix=a.prefix)
    if a.mse_weight:
        trainer.loss = MSEAugmentedLoss(trainer.loss, a.mse_weight)
    if a.full:
        from experiments.n1b_native.data import install_coffee_cache
        from Datasets.PlenopticVideoBlender import CustomDataset
        verified = install_coffee_cache(ROOT)
        dataset = CustomDataset(config['DATASET']['PATH'])
        record['dataset'] = {'train_views': len(dataset.train()), 'test_views': len(dataset.test()),
            'metadata_sha256': {name: sha(dataset.dataset_path/name) for name in
                ('transforms_train.json','transforms_test.json','points3d.ply')}}
        assert record['dataset']['train_views'] == 5100 and record['dataset']['test_views'] == 300
    else:
        dataset = DevelopmentDataset(config['DATASET']['PATH'])
        dataset.preload(secondary=a.secondary)
        record['selection'] = dataset.selection
        record['pixel_hash'] = dataset.pixel_hash
    if a.checkpoint:
        record['shared_prefix'] = resume(trainer, dataset, a.checkpoint, a.kind, a.accel_lr, a.color_lr)
        record['prefix_sha256'] = sha(a.checkpoint)
        # A no-update equivalence audit, measured against repeated native output.
        view = dataset.train()[0]
        with torch.no_grad():
            base = FasterGS4DRenderer(model)
            x, _ = base.render_image_training(view, False, view.camera.background_color)
            y, _ = renderer.render_image_training(view, False, view.camera.background_color)
            z, _ = base.render_image_training(view, False, view.camera.background_color)
            error = float((x-y).square().mean().sqrt())
            repeat = float((x-z).square().mean().sqrt())
            record['zero_residual_audit'] = {'rmse': error, 'native_repeat_rmse': repeat}
            (out/'zero_residual_audit.json').write_text(json.dumps(record['zero_residual_audit']))
            assert error <= 3*repeat + 1e-7, record['zero_residual_audit']
    torch.cuda.synchronize()
    begin = trainer.started = time.perf_counter()
    trainer.run(dataset)
    torch.cuda.synchronize()
    record['seconds'] = time.perf_counter()-begin
    record['steps'] = model.num_iterations_trained
    record['final_gaussians'] = len(model.gaussians.spatial_means)
    record['callbacks'] = {f.__name__: f.timer.get_values()[2] for f in trainer._list_callbacks()
                          if hasattr(f.timer,'get_values')}
    assert record['steps'] == a.steps
    if not a.prefix:
        state = {'raw': raw_state(model),
            'active_sh_degree': model.gaussians.active_sh_degree,
            'steps': model.num_iterations_trained,
            'residuals': {k:v.detach().cpu() for k,v in model.gaussians.residuals.items()},
            'sigma_t': None if model.gaussians.reference_sigma_t is None else model.gaussians.reference_sigma_t.cpu(),
            'sigma_xyz': None if model.gaussians.reference_sigma_xyz is None else model.gaussians.reference_sigma_xyz.cpu()}
        assert all(bool(v.isfinite().all()) for v in [*state['raw'].values(),*state['residuals'].values()])
        torch.save(state, out/'final.pt')
        record['checkpoint_sha256'] = sha(out/'final.pt')
        (out/'training.json').write_text(json.dumps(record, indent=2))
        views = dataset.test() if a.full else (dataset.secondary if a.secondary else dataset.data['val'])
        record['evaluation_split'] = 'official_test' if a.full else ('reserved' if a.secondary else 'primary')
        if a.full:
            record['cache_checks'] = len(verified)
        record['validation'] = evaluate(renderer, views)
    else:
        record['checkpoint_sha256'] = sha(out/'prefix.pt')
    (out/'results.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'complete': str(out), 'metrics': record.get('validation',{}).get('mean')}), flush=True)


if __name__ == '__main__':
    main()
