"""Predeclared second development round, same prefix and primary camera."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ARMS = [
    ('accel_color', ['--kind','accel_color']),
    ('accel_fast', ['--kind','accel','--accel-lr','.05']),
    ('color_fast', ['--kind','color','--color-lr','.005']),
    ('mse2', ['--kind','native','--mse-weight','2']),
    ('mse8', ['--kind','native','--mse-weight','8']),
    ('accel_mse2', ['--kind','accel','--mse-weight','2']),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arms', nargs='*')
    a = p.parse_args()
    root = Path('output/n1b_v4')
    prefix = root/f'dev_s{a.seed}_prefix/prefix.pt'
    assert prefix.exists()
    for name, args in ARMS:
        if a.arms and name not in a.arms:
            continue
        out = root/f'dev_s{a.seed}_{name}'
        if (out/'results.json').exists():
            continue
        if out.exists():
            raise RuntimeError(f'Incomplete run preserved: {out}')
        print(json.dumps({'starting':str(out)}), flush=True)
        with (root/(out.name+'_round2.log')).open('w') as log:
            r = subprocess.run([sys.executable,'-u','-X','faulthandler','-m','experiments.n1b_v4.run',
                '--seed',str(a.seed),'--checkpoint',str(prefix),'--out',str(out),*args],
                stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f'{out}: exit {r.returncode}')
        record = json.loads((out/'results.json').read_text())
        print(json.dumps({'completed':str(out),'validation':record['validation']['mean']}), flush=True)

if __name__ == '__main__':
    main()
