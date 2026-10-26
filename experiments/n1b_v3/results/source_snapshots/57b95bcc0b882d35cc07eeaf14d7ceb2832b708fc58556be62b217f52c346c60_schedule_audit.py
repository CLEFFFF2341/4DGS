"""Fixed-model real training-observation comparison of bounded partitions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.n1b.core import render_cuda
from experiments.n1b_v2.real_dynamic import DynamicData
from .gradient_audit import estimate
from .support_schedule import schedule


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);rows=[]
    for run in ['output/dsec_c120_s0','output/dsec_n120_s0']:
        args=json.loads((Path(run)/'results.json').read_text())['args']
        data=DynamicData(args['root'],args['start'],args['count'],args['width'])
        for state in ['shared_initial','gauss']:
            initial=torch.load(Path(run)/(state+'.pt'),weights_only=True)
            for side,index in [data.train[j] for j in (0,11,24,35)]:
                ref,refgrad,refloss,_=estimate(initial,data,side,index,64)
                coarse,coarsegrad,_,base_seconds=estimate(initial,data,side,index,8)
                base_image=float((ref-coarse).square().mean())
                base_grad=sum(float((refgrad[k]-coarsegrad[k]).double().square().sum()) for k in refgrad)
                row={'run':run,'state':state,'side':side,'frame':data.frames[index],
                     'gauss8':{'image_mse':base_image,'gradient_square_error':base_grad,'seconds':base_seconds},'proposals':{}}
                for method in ('weighted_edges','top_edges'):
                    raw={k:v.detach().clone().requires_grad_() for k,v in initial.items()}
                    torch.cuda.synchronize();start=time.perf_counter()
                    nodes,info=schedule(raw,data.cameras[side],*data.bounds[side,index],8,method)
                    pred=sum(float(w)*render_cuda(raw,data.cameras[side],float(t),data.bg) for t,w in nodes)
                    loss=(pred-data.images[side][index]).square().mean()
                    grad=torch.autograd.grad(loss,tuple(raw.values()))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    error=float((pred.detach()-ref).square().mean())
                    gerror=sum(float((g-refgrad[k]).double().square().sum()) for k,g in zip(raw,grad))
                    row['proposals'][method]={'image_mse':error,'gradient_square_error':gerror,
                                              'image_ratio':error/max(base_image,1e-30),
                                              'gradient_ratio':gerror/max(base_grad,1e-30),
                                              'seconds':elapsed,'schedule':info}
                rows.append(row);path.write_text(json.dumps({'rows':rows},indent=2))
                print(json.dumps({'run':run,'state':state,'frame':data.frames[index],
                                  'ratios':{k:[v['image_ratio'],v['gradient_ratio']] for k,v in row['proposals'].items()}}),flush=True)
    summary={m:{'image_wins':sum(r['proposals'][m]['image_ratio']<1 for r in rows),
                'gradient_wins':sum(r['proposals'][m]['gradient_ratio']<1 for r in rows),
                'median_image_ratio':float(np.median([r['proposals'][m]['image_ratio'] for r in rows])),
                'median_gradient_ratio':float(np.median([r['proposals'][m]['gradient_ratio'] for r in rows])),
                'median_time_ratio':float(np.median([r['proposals'][m]['seconds']/r['gauss8']['seconds'] for r in rows]))}
             for m in ('weighted_edges','top_edges')}
    path.write_text(json.dumps({'rows':rows,'summary':summary},indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
