"""Independent floating-point LPIPS and quadrature audit for saved new models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from experiments.n1b_v2.real_dynamic import DynamicData
from .models import predict_state


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arms',nargs='+',default=['native:1:0','affine_spline:1:0.00003'])
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    net=LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True).cuda().eval();rows=[]
    for run in a.runs:
        folder=Path(run);result=json.loads((folder/'results.json').read_text());args=result['previous_args']
        data=DynamicData(args['root'],args['start'],args['count'],args['width'])
        row={'run':run,'sequence':data.root.name,'start':data.start,'seed':args['seed'],'arms':{}}
        for arm in a.arms:
            path=folder/(arm.replace(':','_')+'.pt');state=torch.load(path,weights_only=True)
            lp=[];quad=[];float_metrics=[]
            for s,i in data.test:
                image=predict_state(state,data,s,i,32);fine=predict_state(state,data,s,i,64)
                quad.append(float((image-fine).square().mean()))
                lp.append(float(net(image[None].clamp(0,1),data.images[s][i][None])));net.reset()
                float_metrics.append(-10*np.log10(float((image.clamp(0,1)-data.images[s][i]).square().mean())))
            row['arms'][arm]={'lpips':float(np.mean(lp)),'per_view_lpips':lp,
                              'psnr_recomputed':float(np.mean(float_metrics)),
                              'quadrature_mse_32_vs_64':float(np.mean(quad)),
                              'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                              'hash_matches_training_record':hashlib.sha256(path.read_bytes()).hexdigest()==result['arms'][arm]['checkpoint_sha256']}
        rows.append(row);out.write_text(json.dumps({'rows':rows},indent=2));print(json.dumps(row),flush=True)


if __name__=='__main__':main()
