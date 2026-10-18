"""Validate final experiment artifacts and write a reviewable result record."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    check = json.loads((ROOT / 'deployment/final_checkpoint_validation.json').read_text())
    assert check['num_iterations_trained'] == 30000 and check['all_floating_parameters_finite']
    run = Path(check['run_directory'])
    renders = run / 'inference/test_30000'
    assert len(list((renders / 'rgb').glob('*.png'))) == 300
    assert len(list((renders / 'rgb_gt').glob('*.png'))) == 300
    metrics_path = renders / 'metrics_8bit.txt'
    text = metrics_path.read_text()
    metrics = {name: float(value) for name, value in re.findall(r'([A-Za-z0-9_]+):([-+0-9.eE]+)', text.splitlines()[-1])}
    assert len(metrics) == 3
    image_rows = [line for line in text.splitlines() if re.match(r'^\d+\t', line)]
    assert len(image_rows) == 300
    video = run / 'coffee_martini_cam00_dynamic.mp4'
    ffprobe = next((ROOT / 'tools').rglob('ffprobe.exe'))
    probe = json.loads(subprocess.check_output([str(ffprobe), '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,nb_read_frames,duration',
        '-of', 'json', str(video)], text=True))['streams'][0]
    assert (probe['width'], probe['height']) == (1352, 1014)
    assert probe['r_frame_rate'] == '30/1' and int(probe['nb_read_frames']) == 300
    assert abs(float(probe['duration']) - 10.0) < 0.01
    timing_text = (run / 'timings.txt').read_text()
    training_seconds = float(re.search(r'Time:([0-9.]+)', timing_text).group(1))
    vram_text = (run / 'vram_stats.txt').read_text()
    vram = {name: int(value) for name, value in re.findall(r'(VRAM_\w+):(\d+)', vram_text)}
    result = {'status': 'complete', 'run_directory': str(run), 'iterations': 30000,
        'resolution': [1352, 1014], 'batch_size': 4, 'train_views': 5100, 'test_views': 300,
        'metrics': metrics, 'metrics_path': str(metrics_path), 'training_seconds': training_seconds,
        'final_gaussians': check['parameter_shapes']['gaussians._spatial_means'][0],
        'vram_bytes': vram, 'video': str(video), 'video_probe': probe,
        'dataset': json.loads((ROOT / 'deployment/dataset_settings.json').read_text()),
        'cache_preparation_seconds': json.loads((ROOT / 'deployment/image_cache.json').read_text())['elapsed_seconds'],
        'checkpoint_validation': check}
    result['paper_reproduction_status'] = 'not_verified_single_scene_only'
    result['paper_reproduction_audit'] = 'deployment/PAPER_REPRODUCTION_AUDIT.md'
    (ROOT / 'deployment/results.json').write_text(json.dumps(result, indent=2))
    report = [
        '# coffee_martini 完整实验结果', '',
        '> 全流程已完成；尚不能认定论文各项数值指标已复现。论文表 11 是三个场景的汇总，本次仅 coffee_martini。核查见 PAPER_REPRODUCTION_AUDIT.md。', '',
        f'- 输出目录：`{run}`',
        '- 官方 FasterGS4D 分支：`3dbbb985c5e0f29825d0960563f3c39f29eb3987`',
        '- NeRFICG：`e458ac6e41810ad6d9405687c1e698fadd28dac5`',
        '- 训练：30,000 次迭代，batch size 4，SH degree 3，随机种子 0。',
        '- 分辨率：1352×1014；每相机 300 帧、30 fps。',
        '- 划分：17 路相机共 5100 张训练图；cam00 共 300 张测试图。',
        '- 初始稠密点云：589,789 点；原始视频、图像、相机参数均保留。',
        f'- 最终高斯数量：{result["final_gaussians"]:,}。',
        '- 无损缓存：5400 张 float32 图像全部通过与官方加载器的精确数值相等检查。',
        '', '## 完整测试视角评估', '',
        *[f'- {name}: {value:.6f}' for name, value in metrics.items()],
        '- 指标基于官方 NeRFICG 的 PNG 8-bit 输出计算，包含所有 300 帧。',
        '', '## 耗时与产物', '',
        f'- 官方计时器训练时间：{training_seconds / 60:.2f} 分钟（不含预处理、缓存及后处理）。',
        f'- float32 缓存准备时间：{result["cache_preparation_seconds"]:.2f} 秒。',
        f'- 训练峰值显存：allocated {vram["VRAM_allocated"] / 1024**3:.2f} GiB；reserved {vram["VRAM_reserved"] / 1024**3:.2f} GiB。',
        f'- 测试指标：`{metrics_path}`',
        f'- 动态视频：`{video}`',
        '- 视频已解码核验：1352×1014，300 帧，30 fps，10 秒。',
        '- 最终检查点已核验：30000 次迭代，所有浮点参数有限。',
        '', '环境版本、Windows 编译适配和 LPIPS 权重来源详见同目录 README.md。',
    ]
    (ROOT / 'deployment/RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'metrics', 'training_seconds', 'video']}, indent=2))

if __name__ == '__main__':
    main()
