"""Fetch the timm publication of the original torchvision VGG16 weights for LPIPS."""
import concurrent.futures
import hashlib
import json
from pathlib import Path

import download_dataset as downloader

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'b8d8aa2dd860af9233c8c67385a8097fd6c35d3f'
SHA256 = '1d677a45f252c18f796d56251b7584e9fdc977dc3053ea0b2cd25d519f5d0fb2'
downloader.URL = f'https://huggingface.co/timm/vgg16.tv_in1k/resolve/{COMMIT}/pytorch_model.bin'
downloader.SIZE = 553441233
downloader.PARTS = ROOT / 'downloads/vgg16.parts'
downloader.PARTS.mkdir(parents=True, exist_ok=True)
width = (downloader.SIZE + 31) // 32
jobs = [(i, i * width, min(downloader.SIZE - 1, (i + 1) * width - 1)) for i in range(32)]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(downloader.fetch, jobs))
sha256 = hashlib.sha256()
target = ROOT / 'downloads/vgg16-timm-torchvision.bin'
with target.open('wb') as output:
    for i, _, _ in jobs:
        with (downloader.PARTS / f'{i:03d}.part').open('rb') as source:
            while block := source.read(1024 * 1024):
                output.write(block)
                sha256.update(block)
if sha256.hexdigest() != SHA256:
    raise RuntimeError('VGG16 checkpoint SHA256 mismatch')
(ROOT / 'deployment/vgg16_archive.json').write_text(json.dumps({
    'url': downloader.URL, 'sha256': SHA256, 'size': downloader.SIZE,
    'model_card': 'https://huggingface.co/timm/vgg16.tv_in1k',
    'description': 'Original torchvision VGG16 ImageNet-1K weights, republished by timm.'
}, indent=2))
print('VGG16 checkpoint verified', flush=True)
