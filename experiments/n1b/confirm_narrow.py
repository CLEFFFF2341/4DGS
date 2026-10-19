"""Independent 128-seed confirmation of the narrow aligned pilot candidate."""
import json
from pathlib import Path
import numpy as np
import torch
from .narrow_benchmark import fixture,reference,timed,calibrate


def main():
    torch.set_num_threads(4)
    out={'budget_seconds':.05,'seeds':'10000..10127, independent of pilot','repeats':128,
         'reason':'12 pilot seeds may miss rare large residuals for sigma=.001; confirm both aligned phases before claiming benefit',
         'cases':{}}
    for phase in [-.003,.003]:
        raw,cam,pix,bg=fixture(.001,phase,False)
        raw={k:v.requires_grad_() for k,v in raw.items()}
        gen=torch.Generator(device='cuda').manual_seed(671)
        weight=torch.randn((64,3),generator=gen,device='cuda',dtype=torch.double)/192
        ref,grad,n=reference(raw,cam,pix,bg,weight,8)
        case={}
        for gradient in [False,True]:
            rows={}
            for method in ['stratified','gauss','pulse','single','anchored']:
                k=max(1,int(.8*calibrate(method,.05,raw,cam,pix,bg,weight,gradient)))
                trials=[]
                for seed in range(10000,10128):
                    torch.manual_seed(seed)
                    value,elapsed=timed(method,k,raw,cam,pix,bg,weight,gradient)
                    target=grad if gradient else ref
                    trials.append({'seconds':elapsed,'mse':float((value-target).square().mean())})
                errors=np.array([t['mse'] for t in trials])
                rows[method]={'calls':k,'mean_mse':float(errors.mean()),'median_mse':float(np.median(errors)),
                    'max_mse':float(errors.max()),'median_seconds':float(np.median([t['seconds'] for t in trials])),
                    'over_budget_trials':sum(t['seconds']>.05 for t in trials),'trials':trials}
                print(phase,gradient,method,rows[method]['mean_mse'],flush=True)
            rng=np.random.default_rng(317)
            indices=rng.integers(0,128,size=(2048,128))
            cv=np.array([t['mse'] for t in rows['anchored']['trials']])[indices].mean(1)
            best=np.min([np.array([t['mse'] for t in rows[m]['trials']])[indices].mean(1)
                         for m in ['stratified','gauss','pulse','single']],axis=0)
            ci=np.quantile(cv/np.maximum(best,1e-300),[.025,.5,.975]).tolist()
            case['gradient' if gradient else 'image']={'methods':rows,'bootstrap_ratio_to_best_simple_95pct':ci}
            out['cases'][str(phase)]=case
            Path('experiments/n1b/results/narrow_confirmation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    out['decision']='BENEFIT_CONFIRMED_REVIEW_STEP_2_3' if all(
        c[m]['bootstrap_ratio_to_best_simple_95pct'][2]<.8 and
        all(r['over_budget_trials']==0 for r in c[m]['methods'].values())
        for c in out['cases'].values() for m in ['image','gradient']) else 'STOP_NO_CONFIRMED_JOINT_ADVANTAGE'
    Path('experiments/n1b/results/narrow_confirmation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(out['decision'],flush=True)


if __name__=='__main__': main()
