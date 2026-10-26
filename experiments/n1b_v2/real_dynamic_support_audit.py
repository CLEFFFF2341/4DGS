"""Count final-model shutter-support partitions without rerunning training."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiments.n1b.core import rotation
from experiments.n1b_v2.real_dynamic import DynamicData


@torch.no_grad()
def main():
    root=Path('experiments/n1b_v2/results/real_dynamic');rows=[]
    for path in sorted(root.glob('zurich*_s*.json')):
        result=json.loads(path.read_text());folder=Path('output')/Path(result['args']['out']).name
        data=DynamicData(result['args']['root'],result['args']['start'])
        for name in ('original','modified','gauss'):
            raw=torch.load(folder/(name+'.pt'),weights_only=True)
            R=rotation(raw['left_isoclinic_rotations'],raw['right_isoclinic_rotations'])
            variance=(2*torch.cat((raw['spatial_scales'],raw['temporal_scales']),-1)).exp()
            sigma=(R[:,3].square()*variance).sum(-1).sqrt();radius=math.sqrt(-2*math.log(.05))*sigma
            a=(raw['temporal_means'][:,0]-radius).cpu().numpy().astype(float)
            b=(raw['temporal_means'][:,0]+radius).cpu().numpy().astype(float)
            for side,i in data.test:
                lo,hi=data.bounds[side,i];valid=(a<hi)&(b>lo);starts=np.maximum(a[valid],lo);ends=np.minimum(b[valid],hi)
                cuts=np.unique(np.concatenate(([lo,hi],starts,ends)));mid=(cuts[1:]+cuts[:-1])/2
                active=np.searchsorted(np.sort(starts),mid,side='left')-np.searchsorted(np.sort(ends),mid,side='right')>0
                intervals=int(active.sum())
                rows.append({'run':folder.name,'method':name,'side':side,'frame':data.frames[i],
                             'pre_fallback_intervals':intervals,'whole_shutter_covering_primitives':int(((a<=lo)&(b>=hi)).sum()),
                             'fallback_at_order8':intervals>4,'fallback_at_order32':intervals>16})
    summary={'checked_exposures':len(rows),'intervals_min':min(r['pre_fallback_intervals'] for r in rows),
             'intervals_max':max(r['pre_fallback_intervals'] for r in rows),
             'fallback8_fraction':float(np.mean([r['fallback_at_order8'] for r in rows])),
             'fallback32_fraction':float(np.mean([r['fallback_at_order32'] for r in rows])),
             'rows':rows}
    (root/'support_audit.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='rows'})


if __name__=='__main__':main()
