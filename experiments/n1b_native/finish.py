"""One-shot continuation: finish the frozen queue, audit, report, and publish.

This is not a periodic scheduler. It runs after the existing queue exits and
never overwrites an incomplete training run or changes model settings.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/n1b_native'
DEST = ROOT / 'experiments/n1b_native'
PREFIX = 'native03_'


def call(module, args, log):
    with (OUT / log).open('a') as f:
        result = subprocess.run([sys.executable, '-u', '-m', module, *args],
                                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{module} exited {result.returncode}; see {log}')


def recover_saved_evaluations():
    for folder in sorted(OUT.glob(PREFIX + '*')):
        if not folder.is_dir() or (folder / 'results.json').exists():
            continue
        if (folder / 'training.json').exists() and (folder / 'final.pt').exists():
            call('experiments.n1b_native.recover_eval', [str(folder)], folder.name + '_recovery.log')
        else:
            raise RuntimeError(f'Incomplete training preserved for inspection: {folder}')


def read_pair(dataset, seed):
    return [json.loads((OUT / f'{PREFIX}{dataset}_s{seed}_{arm}/results.json').read_text())
            for arm in ('native', 'field')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--publish', action='store_true')
    a = p.parse_args()
    if a.wait_pid:
        subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            f'Wait-Process -Id {a.wait_pid} -ErrorAction SilentlyContinue'], check=False)
    recover_saved_evaluations()
    # Completed arms are skipped. Fresh missing arms use the original settings.
    for dataset in ('coffee', 'dsec'):
        call('experiments.n1b_native.suite', ['--dataset', dataset, '--prefix', PREFIX],
             f'{PREFIX}{dataset}_finish_suite.log')
    baseline, candidate = read_pair('coffee', 0)
    n, c = [r['metrics']['mean']['midpoint']['8bit'] for r in (baseline, candidate)]
    coffee_gate = c['psnr'] > n['psnr'] and c['ssim'] >= n['ssim'] and c['lpips'] <= n['lpips']
    (OUT / 'coffee_replication_gate.json').write_text(json.dumps({
        'passed': coffee_gate, 'baseline': n, 'candidate': c,
        'rule': 'TRANSFER_GATE.md; frozen before coffee evaluation'}, indent=2))
    if coffee_gate:
        for seed in (1, 2):
            for arm in (('field', 'native') if seed == 1 else ('native', 'field')):
                name = f'{PREFIX}coffee_s{seed}_{arm}'
                folder = OUT / name
                if (folder / 'results.json').exists():
                    continue
                if folder.exists():
                    raise RuntimeError(f'Incomplete replication preserved: {folder}')
                call('experiments.n1b_native.run', ['--dataset', 'coffee', '--seed', str(seed),
                     '--arm', arm, '--out', str(folder)], name + '.log')
    call('experiments.n1b_native.assess', [], 'final_assessment.log')
    report = json.loads((DEST / 'results/assessment.json').read_text())
    pairs = [r for r in report['pairs'] if r['name'].startswith(PREFIX)]
    dsec = [r for r in pairs if r['dataset'] == 'dsec']
    coffee = [r for r in pairs if r['dataset'] == 'coffee']
    assert len(dsec) == 9 and len(coffee) == (3 if coffee_gate else 1)
    mean = lambda rows, key: sum(r['delta']['midpoint']['8bit'][key] for r in rows) / len(rows)
    clip_gains = {clip: mean([r for r in dsec if f'_{clip}_' in r['name']], 'psnr')
                  for clip in ('c05', 'c08', 'n260')}
    gates = {'complete': True,
        'dsec_positive_psnr_pairs': sum(r['delta']['midpoint']['8bit']['psnr'] > 0 for r in dsec),
        'dsec_clip_mean_psnr': clip_gains,
        'dsec_passed': sum(r['delta']['midpoint']['8bit']['psnr'] > 0 for r in dsec) >= 8
            and all(v > 0 for v in clip_gains.values())
            and mean(dsec, 'ssim') >= 0 and mean(dsec, 'lpips') <= 0,
        'coffee_seed0_gate_passed': coffee_gate,
        'coffee_three_seed_passed': len(coffee) == 3
            and all(r['delta']['midpoint']['8bit']['psnr'] > 0 for r in coffee)
            and mean(coffee, 'ssim') >= 0 and mean(coffee, 'lpips') <= 0}
    (DEST / 'results/final_gates.json').write_text(json.dumps(gates, indent=2))
    lines = ['# 完整 FasterGS4D 原生对照结果', '',
        '完整结果已生成，并核验检查点哈希、配对初始化和原生训练回调次数。', '',
        f"DSEC 预声明稳定性门槛：{'通过' if gates['dsec_passed'] else '未通过'}。",
        f"Coffee 首组转移门槛：{'通过' if coffee_gate else '未通过'}。",
        f"Coffee 三种子复测：{'通过' if gates['coffee_three_seed_passed'] else ('未通过' if coffee_gate else '首组未通过，按预声明规则未扩展')}。", '',
        '这些结论限于此处的数据、实现和预算；一个场景的多种子结果不能证明跨场景稳定优势。', '',
        '## 单时刻推理，官方 8-bit 量化，VGG LPIPS', '',
        '| 配对 | 原生 PSNR | 候选 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS（低更好） |',
        '|---|---:|---:|---:|---:|---:|']
    for r in pairs:
        n, c = [r['arms'][arm]['metrics']['midpoint']['8bit'] for arm in ('native', 'field')]
        d = r['delta']['midpoint']['8bit']
        lines.append(f"| {r['name']} | {n['psnr']:.6f} | {c['psnr']:.6f} | {d['psnr']:+.6f} | {d['ssim']:+.6f} | {d['lpips']:+.6f} |")
    lines += ['', '## 训练成本与容量', '',
        '每臂均训练 30,000 步。DSEC batch 1，Coffee batch 4。相同迭代数不代表相同耗时或最终容量。', '',
        '| 配对 | 原生秒数 | 候选秒数 | 原生高斯数 | 候选高斯数 |',
        '|---|---:|---:|---:|---:|']
    for r in pairs:
        n, c = [r['arms'][arm] for arm in ('native', 'field')]
        lines.append(f"| {r['name']} | {n['training_seconds']:.2f} | {c['training_seconds']:.2f} | {n['gaussians']} | {c['gaussians']} |")
    lines += ['', '## 范围与核验', '',
        '- DSEC 为相机支架坐标下的自定义窄基线双目适配，不能视为官方 DSEC 基准。',
        '- Coffee 使用原生 PlenopticVideoBlender 加载器、1352×1014 分辨率、5,100 张训练图、全部 300 张测试图和同一点云。',
        '- 候选为冻结的 8 时间节点、96 参数全局仿射中心场，学习率 3e-5；保留原生增密、剪枝、重置、SH 升阶、排序及损失。',
        '- 每臂 30,000 次原生训练、144 次增密、4 次训练阶段 Morton 排序；配对初始化哈希一致，最终模型哈希已验证。',
        '- 原生随机采样器不变，增密造成的随机数消耗差异可能使后续采样次序不同。',
        '- 训练时间为原生训练及密度等训练回调累计，不含数据准备、评估和导出；不同容量也会影响耗时。',
        '- 所有逐视图指标、Gauss32 补充结果及来源哈希见 results/；中断及恢复过程见 RUN_LOG.md。',
        '- 失败和中断尝试保留在 output/n1b_native；未完整训练或缺失评估的尝试不计为成功实验。', '']
    (DEST / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    # Retain fixed, previously selected views, without selecting by outcome.
    comparisons = DEST / 'results/comparisons'
    comparisons.mkdir(exist_ok=True)
    for name, view in [('c05_s3', 6), ('c08_s3', 6), ('n260_s3', 6), ('coffee_s0', 150)]:
        for arm in ('native', 'field'):
            folder = OUT / f'{PREFIX}{name}_{arm}'
            for kind in ('midpoint', 'target'):
                path = folder / f'view{view}_{kind}.png'
                if path.exists():
                    shutil.copy2(path, comparisons / f'{name}_{arm}_{kind}.png')
    if a.publish:
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
        assert branch == 'codex/n1b-real-geometry', branch
        subprocess.run(['git', 'add', 'experiments/n1b_native'], cwd=ROOT, check=True)
        # Explicitly include only this experiment's ignored comparison PNGs.
        subprocess.run(['git', 'add', '-f', str(comparisons)], cwd=ROOT, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode:
            subprocess.run(['git', 'commit', '-m', 'Record complete native FasterGS4D transfer results'], cwd=ROOT, check=True)
        subprocess.run(['git', 'push', 'origin', 'codex/n1b-real-geometry'], cwd=ROOT, check=True)
    status = {'complete': True, 'finished_unix': time.time(), 'gates': gates,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (OUT / 'completion.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (OUT / 'completion_error.json').write_text(json.dumps({
            'complete': False, 'error': str(error), 'time': time.time()}, indent=2))
        raise
