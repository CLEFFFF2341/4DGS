"""Paired fresh-seed runs for a fixed candidate; never auto-select a candidate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--kind', required=True)
    p.add_argument('--mse-weight', type=float, default=0.)
    p.add_argument('--accel-lr', type=float, default=.01)
    p.add_argument('--color-lr', type=float, default=.001)
    p.add_argument('--full', action='store_true')
    a = p.parse_args()
    root = Path('output/n1b_v4')
    tag = f'{"full" if a.full else "dev"}_s{a.seed}'
    prefix = root/f'{tag}_prefix'
    params = ['--kind',a.kind,'--mse-weight',str(a.mse_weight),
              '--accel-lr',str(a.accel_lr),'--color-lr',str(a.color_lr)]
    jobs = [(prefix,['--prefix','--steps','15000']),
            (root/f'{tag}_native',['--checkpoint',str(prefix/'prefix.pt')]),
            (root/f'{tag}_candidate',['--checkpoint',str(prefix/'prefix.pt'),*params])]
    if a.kind != 'native' and a.mse_weight:
        jobs.append((root/f'{tag}_loss_control', ['--checkpoint',str(prefix/'prefix.pt'),
                     '--mse-weight',str(a.mse_weight)]))
    declaration = root/f'{tag}_confirmation_plan.json'
    text = json.dumps(vars(a),indent=2)
    if declaration.exists():
        assert declaration.read_text() == text, 'Candidate differs from frozen plan'
    else:
        declaration.write_text(text)
    for out, args in jobs:
        if (out/'results.json').exists():
            print(json.dumps({'already_completed':str(out)}),flush=True)
            continue
        if out.exists():
            raise RuntimeError(f'Incomplete run preserved: {out}')
        print(json.dumps({'starting':str(out)}),flush=True)
        with (root/(out.name+'.log')).open('w') as log:
            result = subprocess.run([sys.executable,'-u','-X','faulthandler','-m','experiments.n1b_v4.run',
                '--seed',str(a.seed),'--out',str(out),*(['--full'] if a.full else []),*args],
                stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{out}: exit {result.returncode}')
        r = json.loads((out/'results.json').read_text())
        print(json.dumps({'completed':str(out),'validation':r.get('validation',{}).get('mean')}),flush=True)

if __name__ == '__main__':
    main()
