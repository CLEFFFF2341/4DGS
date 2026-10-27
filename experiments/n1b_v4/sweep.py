"""Serial real-data development; evaluation only on the predeclared inner split."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    root = Path('output/n1b_v4')
    prefix = root/f'dev_s{a.seed}_prefix'
    jobs = [(prefix, ['--prefix','--steps','15000'])]
    for kind in ('native','affine','accel','color','accel_color'):
        jobs.append((root/f'dev_s{a.seed}_{kind}', ['--checkpoint',str(prefix/'prefix.pt'),'--kind',kind]))
    for out, args in jobs:
        if (out/'results.json').exists():
            print(json.dumps({'already_completed':str(out)}), flush=True)
            continue
        if out.exists():
            raise RuntimeError(f'Incomplete run preserved: {out}')
        print(json.dumps({'starting':str(out)}), flush=True)
        with (root/(out.name+'.log')).open('w') as log:
            result = subprocess.run([sys.executable,'-u','-m','experiments.n1b_v4.run',
                '--seed',str(a.seed),'--out',str(out),*args],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{out}: exit {result.returncode}')
        record = json.loads((out/'results.json').read_text())
        print(json.dumps({'completed':str(out),'validation':record.get('validation',{}).get('mean')}),flush=True)


if __name__ == '__main__':
    main()
