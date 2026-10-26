"""Aggregate frozen real runs and independently audit perception/integration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from experiments.n1b.core import render_cuda
from experiments.n1b_v2.real_cdavis import Dataset, integrated


def main():
    destination=Path('experiments/n1b_v2/results/real_cdavis')
    destination.mkdir(parents=True,exist_ok=True)
    runs=[]
    for path in sorted(Path('output').glob('real_cdavis_*/results.json')):
        result=json.loads(path.read_text())
        if (result['args']['warmup'],result['args']['steps'],result['args']['points'])==(1200,600,8192): runs.append((path,result))
    assert len(runs)==15, f'Expected 5 scenes x 3 seeds; got {len(runs)}'
    assert len({(r['args']['scene'],r['args']['seed']) for _,r in runs})==15
    perceptual=LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True).cuda().eval()
    audits=[]
    with torch.no_grad():
        for path,result in runs:
            folder=path.parent;d=Dataset(result['args']['root'],result['args']['scene'])
            initial=torch.load(folder/'shared_initial.pt',weights_only=True)
            audit={'scene':result['args']['scene'],'seed':result['args']['seed'],'path':str(folder.resolve()),'arms':{}}
            for name in ['initial','original','modified','gauss']:
                raw=initial if name=='initial' else torch.load(folder/(name+'.pt'),weights_only=True)
                lpips=[]
                for i in d.test:
                    pred=render_cuda(raw,d.cameras[i],d.test_times[i],d.bg).clamp(0,1)
                    lpips.append(float(perceptual(pred[None],d.images[i][None])));perceptual.reset()
                errors=[];reference=[]
                for i in d.train:
                    a,_=integrated(raw,d,i,16,'gauss');b,_=integrated(raw,d,i,32,'gauss')
                    errors.append(float((a-b).square().mean()));reference.append(float((b-d.images[i]).square().mean()))
                displacement=(raw['spatial_means']-initial['spatial_means']).norm(dim=-1)
                entry={'mean_lpips_vgg':float(np.mean(lpips)),'per_view_lpips_vgg':lpips,
                       'mean_quadrature_mse_16_vs_32':float(np.mean(errors)),
                       'train_mse_gauss32':float(np.mean(reference)),
                       'mean_position_displacement_world_units':float(displacement.mean()),
                       'p95_position_displacement_world_units':float(torch.quantile(displacement,.95)),
                       'checkpoint_sha256':hashlib.sha256((folder/('shared_initial.pt' if name=='initial' else name+'.pt')).read_bytes()).hexdigest()}
                audit['arms'][name]=entry
            # Preserve the original run metadata and metrics without changing it.
            (destination/f"{audit['scene']}_s{audit['seed']}.json").write_text(json.dumps(result,indent=2))
            audits.append(audit);print('Audited',audit['scene'],audit['seed'],flush=True)
    aggregate={}
    for name in ['initial','original','modified','gauss']:
        metric=[r['initial'] if name=='initial' else r['arms'][name]['metrics'] for _,r in runs]
        aggregate[name]={'psnr':float(np.mean([m['mean_psnr'] for m in metric])),
                         'ssim':float(np.mean([m['mean_ssim'] for m in metric])),
                         'lpips_vgg':float(np.mean([a['arms'][name]['mean_lpips_vgg'] for a in audits]))}
    (destination/'audit.json').write_text(json.dumps({'aggregate':aggregate,'runs':audits},indent=2))
    print(json.dumps(aggregate),flush=True)


if __name__=='__main__':main()
