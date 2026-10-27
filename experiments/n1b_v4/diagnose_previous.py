"""Describe the already-observed v3 coffee failure; no new test-set queries."""
import json
from pathlib import Path
import numpy as np
import torch


def main():
    root = Path('output/n1b_native')
    n = json.loads((root/'native03_coffee_s0_native/results.json').read_text())
    c = json.loads((root/'native03_coffee_s0_field/results.json').read_text())
    x = torch.load(root/'native03_coffee_s0_field/final.pt',map_location='cpu',weights_only=True)['extra'].double().numpy()
    design = np.stack([np.ones(8),np.linspace(0,1,8)],axis=1)
    linear = design @ np.linalg.lstsq(design,x.reshape(8,-1),rcond=None)[0]
    residual = x.reshape(8,-1)-linear
    rows = []
    for lo in range(0,300,50):
        delta = {key:float(np.mean([c['metrics']['views'][i]['midpoint']['8bit'][key]
                - n['metrics']['views'][i]['midpoint']['8bit'][key] for i in range(lo,lo+50)]))
                for key in ('psnr','ssim','lpips')}
        rows.append({'frames':[lo,lo+49],'delta':delta})
    result = {'source':'previously evaluated frozen v3 seed0; diagnostic only',
              'affine_parameter_norm':float(np.linalg.norm(x)),
              'nonlinear_parameter_energy_fraction':float(np.square(residual).sum()/np.square(x).sum()),
              'per_50_frames':rows,
              'note':'Coefficient energy decomposition is descriptive; it does not establish a causal failure mechanism.'}
    out = Path('experiments/n1b_v4/results');out.mkdir(exist_ok=True)
    (out/'previous_failure_diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
