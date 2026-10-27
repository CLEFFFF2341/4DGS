"""Finish interrupted evaluation from the exact saved final checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from .run import ROOT, Framework, FasterGS4DModel, FasterGS4DRenderer, evaluate, sha
from .native import FieldRenderer
from .data import DSECNativeDataset, install_coffee_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    a = p.parse_args()
    out = Path(a.run).resolve()
    assert not (out / 'results.json').exists()
    record = json.loads((out / 'training.json').read_text())
    assert sha(out / 'final.pt') == record['checkpoint_sha256']
    assert record['steps'] == 30000
    Framework.setup(config_path=out / 'config.yaml')
    torch.set_num_threads(4)
    state = torch.load(out / 'final.pt', map_location='cpu', weights_only=True)
    model = FasterGS4DModel(out.name).build()
    for key, value in state['raw'].items():
        assert bool(value.isfinite().all())
        setattr(model.gaussians, '_' + key, torch.nn.Parameter(value.cuda()))
    model.gaussians.active_sh_degree = state['active_sh_degree']
    model.gaussians.active_sh_bases = (state['active_sh_degree'] + 1) ** 2
    model.num_iterations_trained = state['steps']
    if state['extra'] is not None:
        model.temporal_field = torch.nn.Parameter(state['extra'].cuda())
        renderer = FieldRenderer(model)
    else:
        renderer = FasterGS4DRenderer(model)
    args = record['args']
    if args['dataset'] == 'coffee':
        install_coffee_cache(ROOT)
        from Datasets.PlenopticVideoBlender import CustomDataset
        dataset = CustomDataset('dataset/plenoptic_video/coffee_martini')
    else:
        dataset = DSECNativeDataset(args['root'], args['start'], args['seed'])
    record['evaluation_recovery'] = {'source_sha256': sha(__file__),
        'method': 'exact final checkpoint reload; no additional training'}
    record['metrics'] = evaluate(renderer, dataset, out, args['dataset'] == 'dsec')
    (out / 'results.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record['metrics']['mean']), flush=True)


if __name__ == '__main__':
    main()
