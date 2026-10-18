"""Run an unchanged official entry point using verified lossless image caches."""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from Datasets.utils import ImageData, load_image_simple

manifest = json.loads((ROOT / 'deployment/image_cache.json').read_text())
assert manifest['count'] == 5400 and manifest['all_bitwise_equal']
scene = (ROOT / 'dataset/plenoptic_video/coffee_martini/images').resolve()
cache_root = ROOT / '.cache/coffee_martini_float32'
original_load = ImageData._load
verified = set()

def cached_load(self):
    if (self.path.resolve().parent == scene and self.n_channels == 3 and self.channel_offset == 0
            and self.scale_factor == 0.5 and self.data_scale is None and self.load_fn is load_image_simple):
        path = cache_root / (self.path.stem + '.npy')
        tensor = torch.from_numpy(np.load(path, mmap_mode='c', allow_pickle=False))
        assert tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 1014, 1352)
        # Check exact equality again under the entry point's initialized runtime.
        if len(verified) < 8 and self.path.stem not in verified:
            assert torch.equal(tensor, original_load(self)), str(self.path)
            verified.add(self.path.stem)
        return tensor
    return original_load(self)

ImageData._load = cached_load
entry = sys.argv[1]
assert entry in ('scripts/train.py', 'scripts/inference.py')
if entry == 'scripts/inference.py':
    args = sys.argv[2:]
    flag = '-d' if '-d' in args else '--dir'
    run_directory = Path(args[args.index(flag) + 1])
    checkpoint = torch.load(run_directory / 'checkpoints/final.pt', map_location='cpu', weights_only=False)
    assert checkpoint['num_iterations_trained'] == 30000, checkpoint['num_iterations_trained']
    state = checkpoint['model_state_dict']
    assert all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in state.values())
    (ROOT / 'deployment/final_checkpoint_validation.json').write_text(json.dumps({
        'run_directory': str(run_directory), 'num_iterations_trained': checkpoint['num_iterations_trained'],
        'all_floating_parameters_finite': True,
        'parameter_shapes': {name: list(value.shape) for name, value in state.items()},
    }, indent=2))
    del checkpoint, state
sys.argv = sys.argv[1:]
runpy.run_path(str(ROOT / entry), run_name='__main__')
