"""DSEC data adapter and the existing bitwise-equivalent coffee image cache."""
from pathlib import Path
import numpy as np
import torch
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, ImageData, View, load_image_simple
from experiments.n1b_v2.real_dynamic import DynamicData, initialize


class DSECNativeDataset(BaseDataset):
    def __init__(self, root, start, seed, points=32768):
        self.dynamic = DynamicData(root, start, 24, 360)
        self.seed, self.points = seed, points
        super().__init__(root)

    def load(self):
        d = self.dynamic
        cameras = [PerspectiveCamera(shared_settings=self._camera_settings,
                    width=d.w, height=d.h, focal_x=c.fx, focal_y=c.fy,
                    center_x=c.cx, center_y=c.cy) for c in d.cameras]
        subsets = {'train': [], 'test': [], 'val': []}
        for subset, indices in [('train', d.train), ('test', d.test)]:
            for s, i in indices:
                path = d.root / ('left' if s == 0 else 'right') / f'{d.frames[i]:06d}.png'
                rgb = ImageData(path, n_channels=3)
                rgb.set_data(d.images[s][i], to_default_device=True)
                subsets[subset].append(View(cameras[s], s, i, s * 24 + i,
                    torch.linalg.inv(d.cameras[s].w2c.double()).cpu().numpy(),
                    timestamp=float(d.bounds[s, i].mean()), rgb=rgb))
        # Only positions/colors are passed on. Native initialization supplies all
        # temporal parameters, KNN scales, rotations, opacity, and SH itself.
        raw = initialize(d, self.seed, self.points)
        self.point_cloud = BasicPointCloud(raw['spatial_means'],
            raw['sh_coefficients_0'][:, 0] * .28209479177387814 + .5)
        return cameras, subsets


def install_coffee_cache(root):
    root = Path(root)
    scene = (root / 'dataset/plenoptic_video/coffee_martini/images').resolve()
    cache = root / '.cache/coffee_martini_float32'
    original = ImageData._load
    verified = set()

    def cached_load(self):
        if (self.path.resolve().parent == scene and self.n_channels == 3
                and self.channel_offset == 0 and self.scale_factor == .5
                and self.data_scale is None and self.load_fn is load_image_simple):
            result = torch.from_numpy(np.load(cache / (self.path.stem + '.npy'),
                                              mmap_mode='c', allow_pickle=False))
            assert result.dtype == torch.float32 and result.shape == (3, 1014, 1352)
            if len(verified) < 8 and self.path.stem not in verified:
                assert torch.equal(result, original(self)), str(self.path)
                verified.add(self.path.stem)
            return result
        return original(self)

    ImageData._load = cached_load
    return verified
