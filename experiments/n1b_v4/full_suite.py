"""Complete all predeclared full-resolution seeds with the frozen candidate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def wait_for_seed1(root, pid):
    """Read-only process wait on Windows; do not signal or terminate training."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    deadline = time.monotonic()+7200
    try:
        while not (root/'full_s1_candidate/results.json').exists():
            if kernel.WaitForSingleObject(handle, 0) != 258:
                raise RuntimeError('Seed 1 coordinator stopped before candidate results')
            if time.monotonic() > deadline:
                raise TimeoutError('Seed 1 queue wait exceeded two hours')
            time.sleep(10)
    finally:
        kernel.CloseHandle(handle)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[1,2,3])
    p.add_argument('--wait-pid', type=int)
    a = p.parse_args()
    assert len(set(a.seeds)) == len(a.seeds) and all(s in (1,2,3) for s in a.seeds)
    root = Path('output/n1b_v4')
    if a.wait_pid:
        print(json.dumps({'waiting_for_seed1_pid':a.wait_pid}),flush=True)
        wait_for_seed1(root,a.wait_pid)
    for seed in a.seeds:
        r = subprocess.run([sys.executable,'-u','-m','experiments.n1b_v4.confirm',
                            '--full','--seed',str(seed),'--kind','accel','--accel-lr','.05'])
        if r.returncode:
            raise RuntimeError(f'Full seed {seed} stopped: exit {r.returncode}')
    subprocess.run([sys.executable,'-m','experiments.n1b_v4.summarize'],check=True)
    pairs = []
    for seed in (1,2,3):
        paths = [root/f'full_s{seed}_{arm}/results.json' for arm in ('native','candidate')]
        if not all(path.exists() for path in paths):
            continue
        b, c = [json.loads(path.read_text()) for path in paths]
        assert b['prefix_sha256'] == c['prefix_sha256']
        assert c['args']['kind'] == 'accel' and c['args']['accel_lr'] == .05 and c['args']['mse_weight'] == 0
        assert b['args']['kind'] == 'native' and b['args']['mse_weight'] == 0
        assert b['steps'] == c['steps'] == 30000
        assert b['dataset'] == c['dataset']
        assert b['native_source_sha256'] == c['native_source_sha256']
        for key in b['source_sha256']:
            if Path(key).name in ('model.py','train.py','run.py'):
                assert b['source_sha256'][key] == c['source_sha256'][key]
        assert len(b['validation']['views']) == len(c['validation']['views']) == 300
        delta = {k:c['validation']['mean'][k]-b['validation']['mean'][k] for k in ('psnr','ssim','lpips')}
        pairs.append({'seed':seed,'native':b['validation']['mean'],'candidate':c['validation']['mean'],'delta':delta})
    mean = {k:sum(p['delta'][k] for p in pairs)/len(pairs) for k in ('psnr','ssim','lpips')} if pairs else {}
    passed = len(pairs) == 3 and all(p['delta']['psnr'] > 0 for p in pairs) and mean['psnr'] >= .1 and mean['ssim'] >= 0 and mean['lpips'] <= 0
    report = {'scope':'Coffee only; frozen three-seed full native confirmation',
              'complete':len(pairs)==3,'passes_predeclared_gate':passed,'mean_delta':mean,'pairs':pairs}
    Path('experiments/n1b_v4/results/full_confirmation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
