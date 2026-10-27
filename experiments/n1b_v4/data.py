"""Internal camera split; official test pixels are never used for development."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from Datasets.PlenopticVideoBlender import CustomDataset


class DevelopmentDataset(CustomDataset):
    def load(self):
        cameras, data = super().load()
        selected = [v for v in data['train'] if int(v._rgb.path.stem.split('_')[1]) % 10 == 0]
        camera = lambda v: v._rgb.path.stem.split('_')[0]
        primary = [v for v in selected if camera(v) == 'cam08']
        self.secondary = [v for v in selected if camera(v) == 'cam16']
        train = [v for v in selected if camera(v) not in ('cam08', 'cam16')]
        assert len(train) == 450 and len(primary) == len(self.secondary) == 30
        assert all('cam00' not in str(v._rgb.path) for v in train + primary + self.secondary)
        self.selection = {'train': [v._rgb.path.stem for v in train],
                          'validation': [v._rgb.path.stem for v in primary],
                          'reserved': [v._rgb.path.stem for v in self.secondary]}
        return cameras, {'train': train, 'val': primary, 'test': []}

    def preload(self, secondary=False):
        cache = Path('.cache/n1b_v4_native_area_0125')
        cache.mkdir(exist_ok=True)
        views = self.data['train'] + (self.secondary if secondary else self.data['val'])
        records = []
        for i, view in enumerate(views):
            path = cache / (view._rgb.path.stem + '.npy')
            if not path.exists():
                value = view._rgb.image.cpu().numpy()
                np.save(path, value, allow_pickle=False)
            else:
                value = np.load(path, allow_pickle=False)
            assert tuple(value.shape) == (3, view.camera.height, view.camera.width)
            view._rgb.set_data(torch.from_numpy(value), to_default_device=True)
            records.append((view._rgb.path.stem, hashlib.sha256(value.tobytes()).hexdigest()))
            if (i+1) % 100 == 0:
                print(json.dumps({'preloaded': i+1, 'total': len(views)}), flush=True)
        self.pixel_hash = hashlib.sha256(json.dumps(records).encode()).hexdigest()
