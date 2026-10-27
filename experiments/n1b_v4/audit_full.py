"""Verify that all six delivered checkpoints reproduce their measured outputs."""
import json
from pathlib import Path
import subprocess
import sys

def main():
    root = Path('output/n1b_v4')
    audits = []
    for seed in (1,2,3):
        for arm in ('native','candidate'):
            run = root/f'full_s{seed}_{arm}'
            result = json.loads((run/'results.json').read_text())
            check = run/'reload_check.json'
            if not check.exists():
                command = [sys.executable,'-u','-X','faulthandler','-m',
                           'experiments.n1b_v4.evaluate_saved','--run',str(run),'--verify-primary']
                for attempt in range(2):
                    log = root/f'{run.name}.reload_attempt{attempt}.log'
                    with log.open('x') as stream:
                        code = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
                    if code == 0:
                        break
                    # Never retry a numerical mismatch. A startup failure or
                    # crashed evaluation may be retried once on the same file.
                    if 'AssertionError' in log.read_text(errors='replace'):
                        raise RuntimeError(f'Checkpoint reload mismatch: {run}')
                if code:
                    raise RuntimeError(f'Checkpoint audit failed: {run}')
            audit = json.loads(check.read_text())
            assert audit['checkpoint_sha256'] == result['checkpoint_sha256']
            audits.append({'run':run.name,'checkpoint_sha256':audit['checkpoint_sha256'],
                           'mean_delta':audit['reload_mean_delta']})
            print(json.dumps(audits[-1]),flush=True)
    path = Path('experiments/n1b_v4/results/full_checkpoint_audit.json')
    path.write_text(json.dumps(audits,indent=2))

if __name__ == '__main__':
    main()
