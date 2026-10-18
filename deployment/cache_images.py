"""Losslessly cache the official loader's float32 RGB outputs, without quantization."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from Datasets.utils import ImageData

torch.set_num_threads(1)
SCENE = ROOT / 'dataset/plenoptic_video/coffee_martini'
CACHE = ROOT / '.cache/coffee_martini_float32'

def cache(path):
    output = CACHE / (path.stem + '.npy')
    tensor = ImageData(path, n_channels=3, scale_factor=0.5)._load()
    assert tensor.shape == (3, 1014, 1352) and tensor.dtype == torch.float32
    np.save(output, tensor.numpy(), allow_pickle=False)
    reloaded = torch.from_numpy(np.load(output, mmap_mode='c', allow_pickle=False))
    assert torch.equal(tensor, reloaded), str(path)
    return {'image': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cache_bytes': output.stat().st_size, 'bitwise_equal': True}

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = sorted((SCENE / 'images').glob('cam*.png'))
    assert len(paths) == 5400
    start = time.perf_counter()
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(cache, paths):
            records.append(record)
            if len(records) % 100 == 0:
                print(f'Cached and verified {len(records)}/5400', flush=True)
    result = {'source': str(SCENE), 'cache': str(CACHE), 'count': len(records),
              'dtype': 'float32', 'shape': [3, 1014, 1352], 'scale_factor': 0.5,
              'resize_mode': 'official area', 'all_bitwise_equal': True,
              'elapsed_seconds': time.perf_counter() - start,
              'total_bytes': sum(r['cache_bytes'] for r in records), 'images': records}
    (ROOT / 'deployment/image_cache.json').write_text(json.dumps(result, indent=2))
    print(f"Complete: {len(records)} bitwise equal float32 caches, {result['total_bytes']} bytes", flush=True)

if __name__ == '__main__':
    main()
