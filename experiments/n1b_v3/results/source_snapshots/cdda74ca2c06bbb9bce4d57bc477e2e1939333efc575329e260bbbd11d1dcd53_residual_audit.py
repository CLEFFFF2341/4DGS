"""Training-only affine radiometry diagnostic, not a test-image correction."""
import json
from pathlib import Path
import torch
import numpy as np
from experiments.n1b_v2.real_dynamic import DynamicData, predict


@torch.no_grad()
def main():
    output=Path('experiments/n1b_v3/results/radiometry_audit_01.json')
    if output.exists():raise FileExistsError(output)
    allrows=[]
    for run in ['output/dsec_c120_s0','output/dsec_n120_s0']:
        args=json.loads((Path(run)/'results.json').read_text())['args']
        data=DynamicData(args['root'],args['start'],args['count'],args['width'])
        raw=torch.load(Path(run)/'gauss.pt',weights_only=True);rows=[]
        for side,index in data.train:
            x,_=predict(raw,data,side,index,8,'gauss');y=data.images[side][index]
            x=x.reshape(3,-1).double();y=y.reshape(3,-1).double()
            mx=x.mean(-1,keepdim=True);my=y.mean(-1,keepdim=True)
            gain=((x-mx)*(y-my)).mean(-1,keepdim=True)/(x-mx).square().mean(-1,keepdim=True).clamp_min(1e-15)
            bias=my-gain*mx
            before=float((x-y).square().mean());after=float((gain*x+bias-y).square().mean())
            rows.append({'side':side,'frame':data.frames[index],'gain':gain[:,0].tolist(),'bias':bias[:,0].tolist(),
                         'before':before,'after':after,'fraction_reduced':1-after/before})
        allrows.append({'run':run,'rows':rows})
        print(run,'mean training residual reduction',np.mean([r['fraction_reduced'] for r in rows]),flush=True)
    output.write_text(json.dumps(allrows,indent=2))


if __name__=='__main__':main()
