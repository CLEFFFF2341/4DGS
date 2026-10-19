"""Audit existing bounded results; fix pulse-MC timing without expanding experiments."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .benchmark import evaluate_method,quadrature,sync
from .core import render_cuda
from .scenes import scene


def main():
    path=Path('experiments/n1b/results/benchmark.json')
    output=json.loads(path.read_text())
    torch.set_num_threads(4)
    with torch.no_grad():
        for name,record in output['scenes'].items():
            raw,cam,pix,bg=scene(name)
            def render(t): return render_cuda(raw,cam,t,bg)[:,12:20,12:20].permute(1,2,0).reshape(-1,3).double()
            ref=quadrature(render,2049,.3,.7,'simpson')
            values=[]; seconds=[]
            evaluate_method('pulse_mc',raw,cam,pix,bg,render,output['calls']); sync()
            for seed in range(output['repeats']):
                torch.manual_seed(seed+103); sync(); start=time.perf_counter()
                value=evaluate_method('pulse_mc',raw,cam,pix,bg,render,output['calls'])
                sync(); seconds.append(time.perf_counter()-start); values.append(value)
            vv=torch.stack(values); mse=float((vv-ref).square().mean()); elapsed=float(np.median(seconds))
            record['methods']['pulse_mc']={'mse':mse,'mean_bias_rms':float((vv.mean(0)-ref).square().mean().sqrt()),
                'variance':float(vv.var(0).mean()),'median_seconds':elapsed,'mse_times_seconds':mse*elapsed}
            print('rechecked proposal-only MC',name,flush=True)
    output['review_corrections']=['Removed unused pair-integral computation from pulse_mc timing; reran only that comparator on the identical fixtures/seeds. No expanded experiment.']
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')


if __name__=='__main__': main()
