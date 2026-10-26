"""Create a shared training-only initial state without evaluating held-out frames."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from experiments.n1b_v2.real_dynamic import DynamicData, initialize, train


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,default=120)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True)
    p.add_argument('--points',type=int,default=32768);p.add_argument('--warmup',type=int,default=1200)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    data=DynamicData(a.root,a.start,24,360)
    args={**vars(a),'width':360,'count':24,'steps':6400,'calls':1}
    result={'args':args,'dataset':data.manifest(),'role':'train-only initialization; no held-out metric computed',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(result,indent=2))
    raw=initialize(data,a.seed,a.points)
    raw,result['warmup']=train(data,raw,'scale-aware','gauss',a.warmup,8,a.seed,warmup=True)
    path=out/'shared_initial.pt';torch.save(raw,path)
    result['initial_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
