"""Validate the complete official coffee_martini experiment inputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def verify():
    scene = ROOT / 'dataset/plenoptic_video/coffee_martini'
    report = {'image_scale_factor': 0.5, 'width': 1352, 'height': 1014,
              'frames_per_camera': 300, 'fps': 30,
              'max_timestamp_exclusive_seconds': 10.0, 'test_camera': 'cam00'}
    paths = {}
    for split in ('train', 'test'):
        metadata = json.loads((scene / f'transforms_{split}.json').read_text())
        assert metadata['w'] == 2704 and metadata['h'] == 2028
        frames = metadata['frames']
        assert all(0 <= f['time'] < 10.0 for f in frames)
        cameras = sorted({Path(f['file_path']).name[:5] for f in frames})
        for camera in cameras:
            times = sorted(f['time'] for f in frames if Path(f['file_path']).name.startswith(camera))
            assert len(times) == 300 and np.allclose(times, np.arange(300) / 30.0)
        assert len(frames) == (5100 if split == 'train' else 300)
        assert (cameras == ['cam00']) if split == 'test' else ('cam00' not in cameras and len(cameras) == 17)
        paths[split] = {f['file_path'] for f in frames}
        for frame in frames:
            with Image.open(scene / (frame['file_path'] + '.png')) as image:
                assert image.size == (2704, 2028)
        report[split] = {'views': len(frames), 'cameras': cameras}
    assert not paths['train'].intersection(paths['test'])
    with (scene / 'points3d.ply').open('rb') as cloud:
        header = []
        for _ in range(100):
            line = cloud.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        else:
            raise RuntimeError('Invalid PLY header')
    count = int(next(line.split()[-1] for line in header if line.startswith('element vertex ')))
    assert count > 0
    report['initial_point_cloud_vertices'] = count
    report['initial_point_cloud_bytes'] = (scene / 'points3d.ply').stat().st_size
    (ROOT / 'deployment/dataset_settings.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    verify()
