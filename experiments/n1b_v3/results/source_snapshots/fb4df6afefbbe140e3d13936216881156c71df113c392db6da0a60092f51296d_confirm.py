"""Execute the frozen confirmatory pairs; no outcome-dependent branches."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
from experiments.n1b_v2.real_dynamic import DynamicData, initialize, train as warmup
from .search import train, evaluate

ROOT=Path('experiments/n1b_v3')
CLIPS=[('c05','zurich_city_05_a',120),('c08','zurich_city_08_a',120),('n260','zurich_city_09_a',260)]
ARMS=['native:1:0','affine_spline:1:0.00003']


def check_sources(frozen):
    for path,value in frozen['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=value:
            raise RuntimeError('Frozen source changed: '+path)


def main():
    frozen=json.loads((ROOT/'confirmation01_frozen.json').read_text())
    check_sources(frozen)
    for short,sequence,start in CLIPS:
        for seed in (3,4,5):
            check_sources(frozen)
            data=DynamicData('dataset/dsec_dynamic/'+sequence,start,24,360)
            args={'root':str(data.root),'start':start,'count':24,'width':360,'seed':seed,'warmup':1200,'points':32768}
            initdir=Path('output/n1b_v3/confirm01_initials')/f'{short}_s{seed}'
            initial_path=initdir/'shared_initial.pt'
            if initial_path.exists():
                initial=torch.load(initial_path,weights_only=True)
                initrecord=json.loads((initdir/'initial.json').read_text())
                if initrecord['initial_sha256']!=hashlib.sha256(initial_path.read_bytes()).hexdigest():
                    raise RuntimeError('Initial checkpoint hash mismatch')
            else:
                initdir.mkdir(parents=True,exist_ok=False)
                initial=initialize(data,seed,32768)
                initial,info=warmup(data,initial,'scale-aware','gauss',1200,8,seed,warmup=True)
                torch.save(initial,initial_path)
                initrecord={'args':args,'dataset':data.manifest(),'warmup':info,
                            'initial_sha256':hashlib.sha256(initial_path.read_bytes()).hexdigest()}
                (initdir/'initial.json').write_text(json.dumps(initrecord,indent=2))
            for budget in ('renders','time'):
                check_sources(frozen)
                out=Path('output/n1b_v3')/f'confirm01_{short}_s{seed}_{budget}'
                if (out/'COMPLETE').exists():
                    # Resume only verified completed outputs, never overwrite.
                    done=json.loads((out/'results.json').read_text())
                    if done['frozen_protocol']!=frozen or done['initial_sha256']!=initrecord['initial_sha256']:
                        raise RuntimeError('Completed result mismatch')
                    continue
                out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
                for path in ROOT.glob('*.py'):shutil.copy2(path,out/'source'/path.name)
                result={'previous_args':args,'dataset':data.manifest(),'role':'frozen confirmation 01',
                        'budget':budget,'frozen_protocol':frozen,'initial_sha256':initrecord['initial_sha256'],'arms':{}}
                (out/'protocol.json').write_text(json.dumps(result,indent=2))
                order=ARMS if seed%2==0 else list(reversed(ARMS))
                for arm in order:
                    kind,calls,rate=arm.split(':')
                    state,info=train(initial,data,kind,float(rate),int(calls),6400 if budget=='renders' else 30000,
                                     seed+123,None if budget=='renders' else 12.,'core',1200)
                    path=out/(arm.replace(':','_')+'.pt');torch.save(state,path)
                    info['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                    result['arms'][arm]=info
                    (out/'training.json').write_text(json.dumps(result,indent=2))
                for arm,info in result['arms'].items():
                    state=torch.load(out/(arm.replace(':','_')+'.pt'),weights_only=True)
                    info['metrics']=evaluate(state,data,32,out/arm.replace(':','_'))
                (out/'results.json').write_text(json.dumps(result,indent=2));(out/'COMPLETE').write_text('all fixed arms trained and evaluated\n')
                delta=result['arms'][ARMS[1]]['metrics']['psnr']-result['arms'][ARMS[0]]['metrics']['psnr']
                print(json.dumps({'completed':str(out),'delta_psnr':delta}),flush=True)


if __name__=='__main__':main()
