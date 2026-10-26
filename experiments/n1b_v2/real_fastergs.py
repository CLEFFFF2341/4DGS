"""Fixed-capacity original FasterGS4D core baseline on the frozen DSEC split.

Uses native FusedAdam, unmodified FasterGS4DLoss source and native single-time
rasterization. The full Trainer cannot currently import its MortonEncoding
extension because Windows Application Control blocks that DLL. This adapter
does not change that policy or load/emulate the blocked extension. Densification,
pruning, Morton reordering and SH ramp are excluded for capacity-matched repair
comparison; this is not a full default 30k training reproduction.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from experiments.n1b.core import render_cuda
from experiments.n1b_v2.real_dynamic import DynamicData,evaluate,train

ROOT=Path(__file__).resolve().parents[2]


def original_defaults():
    """Read scalar optimizer/loss defaults without importing the blocked Trainer."""
    tree=ast.parse((ROOT/'src/Methods/FasterGS4D/Trainer.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=='FasterGS4DTrainer':
            configuration=node.decorator_list[0]
            return {kw.arg:{v.arg:ast.literal_eval(v.value) for v in kw.value.keywords}
                    for kw in configuration.keywords if kw.arg in ('OPTIMIZER','LOSS')}
    raise RuntimeError('Original defaults not found')


def load_original_loss(folder):
    sys.path.insert(0,str(ROOT/'src'))
    import Framework
    # The original loss reads only this logging switch. No GUI/model/Trainer is
    # imported, and no blocked Morton extension is substituted.
    Framework.config=Framework.ConfigWrapper.fromDict({'TRAINING':{'WANDB':{'ACTIVATE':False}}})
    path=ROOT/'src/Methods/FasterGS4D/Loss.py'
    spec=importlib.util.spec_from_file_location('original_fastergs_loss',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    defaults=original_defaults()
    # Framework's original GPU runtime constructs metric accumulators on CUDA.
    with torch.device('cuda'):
        loss=module.FasterGS4DLoss(Framework.ConfigParameterList(defaults['LOSS']))
    return loss,defaults


def optimizer(raw,data,defaults):
    from FasterGS4DCudaBackend.torch_bindings.adam import FusedAdam
    # Same extent formula as FasterGS4DTrainer.setup_gaussians; cameras appear
    # equally often in the frozen stereo training observations.
    centers=torch.stack([data.cameras[s].position for s,i in data.train])
    extent=float(1.1*(centers-centers.mean(0)).norm(dim=-1).max())
    d=defaults['OPTIMIZER']
    rates={'spatial_means':d['LEARNING_RATE_MEANS_INIT']*extent,
           'temporal_means':d['LEARNING_RATE_MEANS_INIT']*extent,
           'sh_coefficients_0':d['LEARNING_RATE_SH_COEFFICIENTS_0'],
           'sh_coefficients_rest':d['LEARNING_RATE_SH_COEFFICIENTS_REST'],
           'opacities':d['LEARNING_RATE_OPACITIES'],
           'spatial_scales':d['LEARNING_RATE_SCALES'],'temporal_scales':d['LEARNING_RATE_SCALES'],
           'left_isoclinic_rotations':d['LEARNING_RATE_ROTATIONS'],
           'right_isoclinic_rotations':d['LEARNING_RATE_ROTATIONS']}
    groups=[{'params':[raw[key]],'lr':lr,'name':key} for key,lr in rates.items()]
    return FusedAdam(groups,lr=0.,eps=1e-15),extent,rates


def update_spatial_lr(opt,defaults,extent,iteration):
    d=defaults['OPTIMIZER'];t=min(max(iteration/d['LEARNING_RATE_MEANS_MAX_STEPS'],0.),1.)
    lr=math.exp(math.log(d['LEARNING_RATE_MEANS_INIT']*extent)*(1-t)+math.log(d['LEARNING_RATE_MEANS_FINAL']*extent)*t)
    for group in opt.param_groups:
        if group['name']=='spatial_means':group['lr']=lr


def midpoint(raw,data,side,index):
    t=float(data.bounds[side,index].mean())
    return render_cuda(raw,data.cameras[side],t,data.bg)


@torch.no_grad()
def evaluate_midpoint(raw,data,folder):
    folder.mkdir(parents=True,exist_ok=True);rows=[]
    for s,i in data.test:
        pred=midpoint(raw,data,s,i).clamp(0,1);target=data.images[s][i]
        rows.append({'side':s,'frame':data.frames[i],
                     'psnr':-10*math.log10(float((pred-target).square().mean())),
                     'roi_psnr':-10*math.log10(float((pred[:,data.roi]-target[:,data.roi]).square().mean())),
                     'ssim':float(structural_similarity_index_measure(pred[None],target[None],data_range=1.))})
        Image.fromarray((pred.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(folder/f'{s}_{data.frames[i]:06d}.png')
    return {'views':rows,'mean_psnr':float(np.mean([r['psnr'] for r in rows])),
            'mean_ssim':float(np.mean([r['ssim'] for r in rows])),
            'mean_roi_psnr':float(np.mean([r['roi_psnr'] for r in rows]))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[800,6400]);a=p.parse_args()
    source=Path(a.run);previous=json.loads((source/'results.json').read_text());old=previous['args']
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    data=DynamicData(old['root'],old['start'],old['count'],old['width'])
    initial=source/'shared_initial.pt';raw={k:v.detach().clone().requires_grad_() for k,v in torch.load(initial,weights_only=True).items()}
    loss_fn,defaults=load_original_loss(folder);opt,extent,rates=optimizer(raw,data,defaults)
    result={'source_run':str(source.resolve()),'dataset':data.manifest(),'seed':old['seed'],
            'shared_initial_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),
            'defaults':defaults,'training_cameras_extent':extent,'initial_parameter_rates':rates,
            'schedule_iteration_offset':old['warmup'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'original single-time rasterization + FusedAdam + unmodified FasterGS4DLoss; fixed capacity, no density/Morton/SH scheduling; full Trainer blocked by Windows Application Control',
            'checkpoints':{}}
    paths=['src/Methods/FasterGS4D/Trainer.py','src/Methods/FasterGS4D/Model.py','src/Methods/FasterGS4D/Loss.py','src/Methods/FasterGS4D/Renderer.py']
    result['original_source_sha256']={v:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for v in paths}
    (folder/'protocol.json').write_text(json.dumps(result,indent=2))
    rng=np.random.default_rng(old['seed']+123);order=[]
    while len(order)<max(a.checkpoints):order.extend([data.train[j] for j in rng.permutation(len(data.train))])
    torch.cuda.synchronize();start=time.perf_counter();elapsed=0.
    for step,(s,i) in enumerate(order,1):
        if step>max(a.checkpoints):break
        update_spatial_lr(opt,defaults,extent,old['warmup']+step)
        loss=loss_fn(midpoint(raw,data,s,i),data.images[s][i])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite original FasterGS loss')
        loss.backward();opt.step();opt.zero_grad()
        if step%800==0:print(json.dumps({'step':step,'renders':step,'loss':float(loss.detach()),'seconds':elapsed+time.perf_counter()-start}),flush=True)
        if step in a.checkpoints:
            torch.cuda.synchronize();elapsed+=time.perf_counter()-start
            model={k:v.detach().clone() for k,v in raw.items()};path=folder/f'step{step}.pt';torch.save(model,path)
            result['checkpoints'][str(step)]={'steps':step,'native_training_renders':step,'training_seconds_excluding_checkpoint_io':elapsed,'last_loss':float(loss.detach())}
            start=time.perf_counter()
    # Isolate single-time rendering from the original loss/optimizer changes.
    matched,matched_info=train(data,torch.load(initial,weights_only=True),'scale-aware','gauss',min(a.checkpoints),1,old['seed']+123)
    torch.save(matched,folder/'midpoint_matched.pt')
    result['midpoint_matched_control']=matched_info
    # No test metrics are calculated before all training checkpoints finish.
    for key,entry in result['checkpoints'].items():
        model=torch.load(folder/f'step{key}.pt',weights_only=True)
        entry['midpoint']=evaluate_midpoint(model,data,folder/f'step{key}_midpoint')
        entry['exposure_gauss32']=evaluate(model,data,folder/f'step{key}_exposure')
    result['midpoint_matched_control']['midpoint']=evaluate_midpoint(matched,data,folder/'matched_midpoint')
    result['midpoint_matched_control']['exposure_gauss32']=evaluate(matched,data,folder/'matched_exposure')
    (folder/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({key:{'midpoint':v['midpoint']['mean_psnr'],'exposure':v['exposure_gauss32']['mean_psnr']} for key,v in result['checkpoints'].items()}),flush=True)


if __name__=='__main__':main()
