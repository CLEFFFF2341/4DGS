"""Post-confirmation numerical repeat of the lowest equal-render gain pair.

The case was chosen because it is the minimum, not for candidate selection.
Both repeats and all metrics are retained; no replacement of the original result.
"""
import hashlib
import json
from pathlib import Path
import torch
from experiments.n1b_v2.real_dynamic import DynamicData
from .confirm import ARMS, ROOT, check_sources
from .search import train, evaluate


def main():
    frozen=json.loads((ROOT/'confirmation01_frozen.json').read_text());check_sources(frozen)
    source=Path('output/n1b_v3/confirm01_initials/n260_s4')
    meta=json.loads((source/'initial.json').read_text());args=meta['args']
    initial=torch.load(source/'shared_initial.pt',weights_only=True)
    data=DynamicData(args['root'],args['start'],args['count'],args['width'])
    for repeat in (1,2):
        out=Path('output/n1b_v3')/f'repeat_min_n260_s4_{repeat}';out.mkdir(parents=True,exist_ok=False)
        result={'previous_args':args,'initial_sha256':meta['initial_sha256'],'frozen_protocol':frozen,
                'role':'post-confirmation minimum-gain numerical repeat; not candidate selection','repeat':repeat,'arms':{}}
        (out/'protocol.json').write_text(json.dumps(result,indent=2))
        for arm in ARMS if repeat==1 else reversed(ARMS):
            kind,calls,rate=arm.split(':');state,info=train(initial,data,kind,float(rate),int(calls),6400,127,None,'core',1200)
            path=out/(arm.replace(':','_')+'.pt');torch.save(state,path)
            info['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();result['arms'][arm]=info
        for arm,info in result['arms'].items():
            info['metrics']=evaluate(torch.load(out/(arm.replace(':','_')+'.pt'),weights_only=True),data,32,out/arm.replace(':','_'))
        (out/'results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'repeat':repeat,'delta':result['arms'][ARMS[1]]['metrics']['psnr']-result['arms'][ARMS[0]]['metrics']['psnr']}),flush=True)


if __name__=='__main__':main()
