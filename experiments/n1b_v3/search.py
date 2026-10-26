"""Development-only structure comparisons with immutable per-run outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from experiments.n1b_v2.real_dynamic import DynamicData
from .models import TemporalModel, predict_state


@torch.no_grad()
def evaluate(state, data, calls=32, folder=None):
    rows = []
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=False)
    for s, i in data.test:
        pred = predict_state(state, data, s, i, calls).clamp(0, 1)
        ref = data.images[s][i]
        mse = float((pred-ref).square().mean())
        roi = float((pred[:, data.roi]-ref[:, data.roi]).square().mean())
        rows.append({'side': s, 'frame': data.frames[i], 'psnr': -10*math.log10(mse),
                     'roi_psnr': -10*math.log10(roi),
                     'ssim': float(structural_similarity_index_measure(pred[None], ref[None], data_range=1.))})
        if folder is not None:
            Image.fromarray((pred.permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)).save(folder/f'{s}_{data.frames[i]:06d}.png')
    return {'views': rows, **{k: float(np.mean([r[k] for r in rows])) for k in ('psnr', 'roi_psnr', 'ssim')}}


def train(initial, data, kind, rate, calls, steps, seed, seconds=None, optimizer_name='scale', offset=1200):
    model = TemporalModel(initial, kind=kind, rate=rate, optimizer_name=optimizer_name, data=data)
    loss_fn = None
    if optimizer_name == 'core':
        from experiments.n1b_v2.real_fastergs import load_original_loss, update_spatial_lr
        loss_fn, _ = load_original_loss(None)
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < steps:
        order.extend(data.train[j] for j in rng.permutation(len(data.train)))
    torch.cuda.synchronize()
    start = time.perf_counter(); trajectory = []; count = 0
    for step, (side, index) in enumerate(order[:steps], 1):
        # Time-budget runs stop between completed steps, report overrun explicitly.
        if seconds is not None and step > 1 and time.perf_counter()-start >= seconds:
            break
        model.opt.zero_grad(set_to_none=True)
        if optimizer_name == 'core':
            update_spatial_lr(model.opt, model.defaults, model.extent, offset+step)
        image = model.predict(data, side, index, calls)
        loss = (image-data.images[side][index]).square().mean() if loss_fn is None else loss_fn(image,data.images[side][index])
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        loss.backward(); model.opt.step(); count += 1
        if step == 1 or step % max(200, steps//4) == 0 or step == steps:
            trajectory.append({'step': step, 'loss': float(loss.detach()), 'seconds': time.perf_counter()-start})
            print(json.dumps({'kind': kind, **trajectory[-1]}), flush=True)
        if seconds is not None:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    state = model.state()
    if not all(torch.isfinite(v).all() for v in state['raw'].values()):
        raise RuntimeError('nonfinite model')
    return state, {'steps': count, 'renders': count*calls, 'calls': calls,
                   'training_seconds': elapsed, 'parameter_count': model.parameter_count(),
                   'extra': None if model.extra is None else model.extra.detach().cpu().tolist(),
                   'trajectory': trajectory}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    p.add_argument('--arms', nargs='+', default=['native:8:0', 'anchor:8:0', 'warp:8:0.001', 'curve:8:0.001', 'affine_curve:8:0.0001'])
    p.add_argument('--steps', type=int, default=800); p.add_argument('--seconds', type=float)
    p.add_argument('--eval-calls', type=int, default=32)
    p.add_argument('--optimizer', choices=['scale', 'core'], default='scale')
    a = p.parse_args(); folder = Path(a.out); folder.mkdir(parents=True, exist_ok=False)
    run = Path(a.run); previous = json.loads((run/'results.json').read_text())['args']
    data = DynamicData(previous['root'], previous['start'], previous['count'], previous['width'])
    initial_path = run/'shared_initial.pt'; initial = torch.load(initial_path, weights_only=True)
    sources = ['models.py', 'search.py']
    (folder/'source').mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, folder/'source'/source.name)
    result = {'args': vars(a), 'previous_args': previous, 'dataset': data.manifest(),
              'initial_sha256': hashlib.sha256(initial_path.read_bytes()).hexdigest(),
              'source_sha256': {f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in sources},
              'role': 'development validation; no claim of untouched confirmation', 'arms': {}}
    (folder/'protocol.json').write_text(json.dumps(result, indent=2))
    for arm in a.arms:
        kind, calls, rate = arm.split(':'); calls = int(calls); rate = float(rate)
        name = arm.replace(':', '_')
        state, info = train(initial, data, kind, rate, calls, a.steps, previous['seed']+123, a.seconds, a.optimizer, previous['warmup'])
        torch.save(state, folder/(name+'.pt'))
        info['checkpoint_sha256'] = hashlib.sha256((folder/(name+'.pt')).read_bytes()).hexdigest()
        result['arms'][arm] = info
        (folder/'training.json').write_text(json.dumps(result, indent=2))
    # Never inspect held-out images until all configured training arms finish.
    for arm, info in result['arms'].items():
        name = arm.replace(':', '_')
        info['metrics'] = evaluate(torch.load(folder/(name+'.pt'), weights_only=True), data, a.eval_calls, folder/name)
        (folder/'results.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'arm': arm, 'metrics': {k: v for k, v in info['metrics'].items() if k != 'views'}}), flush=True)


if __name__ == '__main__':
    main()
