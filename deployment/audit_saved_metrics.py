"""Independently check saved PNG PSNR and held-out ground truth provenance."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from Datasets.utils import ImageData
import torch

torch.set_num_threads(1)
result = json.loads((ROOT / 'deployment/results.json').read_text())
folder = Path(result['metrics_path']).parent
predictions = sorted((folder / 'rgb').glob('*.png'))
targets = sorted((folder / 'rgb_gt').glob('*.png'))
assert len(predictions) == len(targets) == 300
values = []
for p, t in zip(predictions, targets):
    assert p.name == t.name
    a = np.asarray(Image.open(p), dtype=np.float64)[..., :3]
    b = np.asarray(Image.open(t), dtype=np.float64)[..., :3]
    assert a.shape == b.shape == (1014, 1352, 3)
    values.append(float(10 * np.log10(255**2 / np.mean((a-b)**2))))
    if len(values) % 100 == 0:
        print(f'Independent PNG PSNR: {len(values)}/300', flush=True)
mean_psnr = float(np.mean(values))
assert abs(mean_psnr-result['metrics']['PSNR']) < 1e-4
scene = ROOT / 'dataset/plenoptic_video/coffee_martini'
frames = json.loads((scene / 'transforms_test.json').read_text())['frames']
samples = []
for index in [0, 150, 299]:
    source = scene / (frames[index]['file_path'] + '.png')
    assert source.stem.startswith('cam00_')
    tensor = ImageData(source, n_channels=3, scale_factor=0.5)._load()
    # Official renderer uses torchvision save_image: multiply 255, add .5, clamp.
    expected = tensor.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    actual = np.asarray(Image.open(targets[index]))[..., :3]
    equal = bool(np.array_equal(expected, actual))
    assert equal, index
    samples.append({'index': index, 'original_source': str(source), 'gt_matches_official_loader_uint8': equal})
report = {'frames': len(values), 'independent_numpy_float64_psnr_mean': mean_psnr,
          'official_psnr_mean': result['metrics']['PSNR'],
          'absolute_difference_db': abs(mean_psnr-result['metrics']['PSNR']),
          'ground_truth_samples': samples, 'per_frame_psnr': values}
(ROOT / 'deployment/metric_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'per_frame_psnr'}, indent=2))
