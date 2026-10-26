"""Predeclared larger-render-budget native Gauss8 controls on reserved seed 3."""
import hashlib
import json
from pathlib import Path
import torch
from experiments.n1b_v2.real_dynamic import DynamicData
from .confirm import CLIPS, ROOT, check_sources
from .search import train,evaluate


def main():
    frozen=json.loads((ROOT/'confirmation01_frozen.json').read_text());check_sources(frozen)
    for short,sequence,start in CLIPS:
        source=Path('output/n1b_v3/confirm01_initials')/f'{short}_s3'
        initial=torch.load(source/'shared_initial.pt',weights_only=True)
        meta=json.loads((source/'initial.json').read_text());data=DynamicData(str(Path('dataset/dsec_dynamic')/sequence),start)
        out=Path('output/n1b_v3')/f'gauss8_{short}_s3';out.mkdir(parents=True,exist_ok=False)
        result={'previous_args':meta['args'],'initial_sha256':meta['initial_sha256'],'role':'predeclared Gauss8 control',
                'frozen_protocol':frozen,'arms':{}}
        (out/'protocol.json').write_text(json.dumps(result,indent=2))
        state,info=train(initial,data,'native',0.,8,6400,126,None,'core',1200)
        path=out/'native_8_0.pt';torch.save(state,path);info['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        info['metrics']=evaluate(state,data,32,out/'native_8_0')
        result['arms']['native:8:0']=info;(out/'results.json').write_text(json.dumps(result,indent=2))
        candidate=json.loads((Path('output/n1b_v3')/f'confirm01_{short}_s3_renders'/'results.json').read_text())['arms']['affine_spline:1:0.00003']['metrics']
        print(json.dumps({'clip':short,'gauss8_psnr':info['metrics']['psnr'],'candidate_minus_gauss8':candidate['psnr']-info['metrics']['psnr']}),flush=True)


if __name__=='__main__':main()
