"""Sequential GPU runs; preserve completed arms and every failure."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['coffee', 'dsec'], required=True)
    p.add_argument('--prefix', default='')
    a = p.parse_args()
    root = Path('output/n1b_native')
    root.mkdir(exist_ok=True)
    jobs = []
    if a.dataset == 'coffee':
        jobs = [(f'coffee_s0_{arm}', ['--dataset', 'coffee', '--arm', arm, '--seed', '0'])
                for arm in ('native', 'field')]
    else:
        for short, scene, start in [('c05', 'zurich_city_05_a', 120),
                                     ('c08', 'zurich_city_08_a', 120),
                                     ('n260', 'zurich_city_09_a', 260)]:
            for seed in (3, 4, 5):
                for arm in (('native', 'field') if seed != 4 else ('field', 'native')):
                    jobs.append((f'{short}_s{seed}_{arm}', [
                        '--dataset', 'dsec', '--root', f'dataset/dsec_dynamic/{scene}',
                        '--start', str(start), '--seed', str(seed), '--arm', arm]))
    for name, args in jobs:
        name = a.prefix + name
        out = root / name
        if (out / 'results.json').exists():
            print(json.dumps({'already_complete': name}), flush=True)
            continue
        if out.exists():
            raise RuntimeError(f'Incomplete run preserved; inspect before continuing: {out}')
        command = [sys.executable, '-m', 'experiments.n1b_native.run', *args, '--out', str(out)]
        started = time.time()
        print(json.dumps({'starting': name, 'time': started}), flush=True)
        with (root / (name + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status = {'name': name, 'command': command, 'returncode': result.returncode,
                  'elapsed_seconds': time.time() - started}
        with (root / 'suite_events.jsonl').open('a') as f:
            f.write(json.dumps(status) + '\n')
        print(json.dumps(status), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
