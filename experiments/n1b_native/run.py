"""One independent process per arm, with the full native 30k-step schedule."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
import Framework
from Datasets.utils import save_image
from Methods.FasterGS4D.Model import FasterGS4DModel
from Methods.FasterGS4D.Renderer import FasterGS4DRenderer
from .native import AuditedNativeTrainer, FieldTrainer, FieldRenderer, raw_state
from .data import DSECNativeDataset, install_coffee_cache


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(renderer, dataset, out, dsec):
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda().eval()
    renderer.model.eval()
    rows = []
    for j, view in enumerate(dataset.test()):
        target = view.rgb.cuda()
        pred = renderer.render_image_inference(view, to_chw=True)['rgb']
        methods = {'midpoint': pred}
        if dsec:
            lo, hi = dataset.dynamic.bounds[view.camera_index, view.frame_idx]
            nodes, weights = np.polynomial.legendre.leggauss(32)
            integrated = torch.zeros_like(pred)
            timestamp = view.timestamp
            for node, weight in zip(nodes, weights):
                view.timestamp = float((lo + hi) / 2 + node * (hi - lo) / 2)
                # Integrate before clamping, as in the previous Gauss32 checks.
                sample, _ = renderer.render_image_training(view, False, view.camera.background_color)
                integrated.add_(sample, alpha=float(weight / 2))
            view.timestamp = timestamp
            methods['gauss32'] = integrated.clamp(0, 1)
        row = {'frame': view.frame_idx, 'camera': view.camera_index, 'timestamp': view.timestamp}
        for name, image in methods.items():
            row[name] = {}
            for precision in ('float', '8bit'):
                a, b = image, target
                if precision == '8bit':
                    a = a.clamp(0, 1).mul(255).add(.5).byte().float() / 255
                    b = b.clamp(0, 1).mul(255).add(.5).byte().float() / 255
                row[name][precision] = {
                    'psnr': float(-10 * torch.log10((a-b).square().mean())),
                    'ssim': float(structural_similarity_index_measure(a[None], b[None], data_range=1.)),
                    'lpips': float(lp(a[None], b[None]))}
                lp.reset()
            if j in (0, 6, 150):
                save_image(out / f'view{j}_{name}.png', image)
                save_image(out / f'view{j}_target.png', target)
        rows.append(row)
        if (j+1) % 25 == 0:
            print(json.dumps({'eval_views': j+1, 'total': len(dataset)}), flush=True)
    means = {name: {precision: {metric: float(np.mean([r[name][precision][metric] for r in rows]))
                for metric in ('psnr', 'ssim', 'lpips')} for precision in ('float', '8bit')}
             for name in rows[0] if name in ('midpoint', 'gauss32')}
    return {'mean': means, 'views': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['dsec', 'coffee'], required=True)
    p.add_argument('--arm', choices=['native', 'field'], required=True)
    p.add_argument('--root', default='dataset/dsec_dynamic/zurich_city_05_a')
    p.add_argument('--start', type=int, default=120)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--out', required=True)
    p.add_argument('--skip-eval', action='store_true')
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / 'configs/fastergs4d_coffee_martini.yaml').read_text())
    config['GLOBAL'].update(RANDOM_SEED=args.seed, LOG_LEVEL=0)
    training = config['TRAINING']
    training.update(NUM_ITERATIONS=args.steps, MODEL_NAME=out.name)
    training['BACKUP'].update(FINAL_CHECKPOINT=False, TRAINING_STATE=False)
    if args.dataset == 'dsec':
        config['GLOBAL']['DATASET_TYPE'] = 'DSECNativeAdapter'
        training['BATCH_SIZE'] = 1
        config['DATASET'].update(PATH=args.root, IMAGE_SCALE_FACTOR=None,
                                 NEAR_PLANE=.05, FAR_PLANE=300.)
    config_path = out / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / p for p in (
        'src/Methods/FasterGS4D/Trainer.py', 'src/Methods/FasterGS4D/Model.py',
        'src/Methods/FasterGS4D/Renderer.py', 'src/Methods/FasterGS4D/Loss.py',
        'src/Methods/Base/Trainer.py', 'src/Optim/Samplers/DatasetSamplers.py',
        'src/Optim/Samplers/utils.py', 'experiments/n1b_v2/real_dynamic.py')]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    record = {'args': vars(args), 'source_sha256': sources,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (out / 'protocol.json').write_text(json.dumps(record, indent=2))
    Framework.setup(config_path=config_path)
    torch.set_num_threads(4)
    verified = set()
    if args.dataset == 'coffee':
        verified = install_coffee_cache(ROOT)
        from Datasets.PlenopticVideoBlender import CustomDataset
        dataset = CustomDataset(config['DATASET']['PATH'])
        record['dataset'] = {'train_views': len(dataset.train()), 'test_views': len(dataset.test()),
                            'metadata_sha256': {name: sha(dataset.dataset_path / name) for name in
                                ('transforms_train.json', 'transforms_test.json', 'points3d.ply')}}
    else:
        dataset = DSECNativeDataset(args.root, args.start, args.seed)
        record['dataset'] = dataset.dynamic.manifest()
    model = FasterGS4DModel(out.name).build()
    model.output_directory = out
    if args.arm == 'field':
        model.temporal_field = torch.nn.Parameter(torch.zeros((8, 3, 4), device='cuda'))
        renderer = FieldRenderer(model)
        trainer = FieldTrainer(model=model, renderer=renderer)
    else:
        renderer = FasterGS4DRenderer(model)
        trainer = AuditedNativeTrainer(model=model, renderer=renderer)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    trainer.run(dataset)
    torch.cuda.synchronize()
    record['run_seconds_including_setup_cleanup'] = time.perf_counter() - begin
    record.update(initial_sha256=trainer.initial_hash, initial_count=trainer.initial_count,
                  zero_field_check=trainer.zero_field_check,
                  steps=model.num_iterations_trained, final_count=len(model.gaussians.spatial_means),
                  progress=trainer.audit_rows, cache_checks=len(verified))
    record['callback_counts'] = {c.__name__: c.timer.get_values()[2] for c in trainer._list_callbacks()
                                  if hasattr(c.timer, 'get_values')}
    record['callback_seconds'] = {c.__name__: c.timer.get_values()[0] for c in trainer._list_callbacks()
                                  if hasattr(c.timer, 'get_values')}
    state = {'raw': raw_state(model), 'extra': getattr(model, 'temporal_field', None),
             'active_sh_degree': model.gaussians.active_sh_degree, 'steps': model.num_iterations_trained}
    if state['extra'] is not None:
        state['extra'] = state['extra'].detach().cpu()
    assert all(bool(v.isfinite().all()) for v in state['raw'].values())
    assert state['extra'] is None or bool(state['extra'].isfinite().all())
    assert record['steps'] == args.steps
    torch.save(state, out / 'final.pt')
    record['checkpoint_sha256'] = sha(out / 'final.pt')
    (out / 'training.json').write_text(json.dumps(record, indent=2))
    if not args.skip_eval:
        record['metrics'] = evaluate(renderer, dataset, out, args.dataset == 'dsec')
    (out / 'results.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({'completed': str(out), 'metrics': record.get('metrics', {}).get('mean')}), flush=True)


if __name__ == '__main__':
    main()
