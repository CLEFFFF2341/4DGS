"""Windows wrapper around Yang et al.'s official n3v2blender.py.

Only Windows portability and restart support are added. Camera transforms,
frame extraction resolution, time assignment, split and dense COLMAP steps
are retained from the upstream script.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    archive = ROOT / 'downloads/coffee_martini.verified.zip'
    if not (ROOT / 'deployment/dataset_archive.json').exists():
        raise RuntimeError('Run download_dataset.py and verify the archive first.')
    scene = ROOT / 'dataset/plenoptic_video/coffee_martini'
    if not (scene / 'poses_bounds.npy').exists():
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if any('..' in Path(n).parts or Path(n).is_absolute() for n in names):
                raise RuntimeError('Unsafe ZIP member')
            destination = scene.parent if any(n.startswith('coffee_martini/') for n in names) else scene
            destination.mkdir(parents=True, exist_ok=True)
            z.extractall(destination)
    if args.extract_only:
        return
    ffmpeg = next((ROOT / 'tools').rglob('ffmpeg.exe'))
    colmap = next((ROOT / 'tools').rglob('colmap.exe'))
    os.environ['PATH'] = str(ffmpeg.parent) + os.pathsep + str(colmap.parent) + os.pathsep + os.environ['PATH']
    os.environ['QT_PLUGIN_PATH'] = str(colmap.parent.parent / 'plugins')
    probes = []
    for video in sorted(scene.glob('cam*.mp4')):
        probe = subprocess.check_output([str(ffmpeg.with_name('ffprobe.exe')), '-v', 'error',
            '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-of', 'json', str(video)], text=True)
        probes.append({'camera': video.stem, **json.loads(probe)['streams'][0]})
    (ROOT / 'deployment/video_metadata.json').write_text(json.dumps(probes, indent=2))
    source = (ROOT / 'deployment/upstream/n3v2blender.py').read_text()
    source = source.replace("cam_name = video.split('/')[-1].split('.')[-2]", "cam_name = __import__('pathlib').Path(video).stem")
    source = source.replace('f[len(args.path):] for f', "f[len(args.path):].replace(chr(92), '/') for f")
    source = source.replace('array.tostring()', 'array.tobytes()')
    source = source.replace('np.fromstring(blob, dtype=dtype)', 'np.frombuffer(blob, dtype=dtype)')
    # The upstream known-pose model contains one shared PINHOLE camera.
    # Explicitly enforce that contract with modern COLMAP's image reader.
    source = source.replace('colmap feature_extractor ', 'colmap feature_extractor --ImageReader.single_camera 1 ')
    # Avoid symlink privilege requirements on Windows, with identical file bytes.
    source = source.replace('os.symlink(', 'shutil.copy2(')
    # Make frame extraction restartable and noninteractive; keep native resolution and all frames.
    source = source.replace('ffmpeg -i {video}', 'ffmpeg -y -i {video}')
    # Reuse only complete, explicitly verified native-resolution frame sequences.
    complete_cameras = []
    for video in probes:
        camera = video['camera']
        images = sorted((scene / 'images').glob(f'{camera}_*.png'))
        if len(images) == int(video['nb_frames']):
            from PIL import Image
            for image in images:
                with Image.open(image) as check:
                    if check.size != (int(video['width']), int(video['height'])):
                        break
                    check.verify()
            else:
                complete_cameras.append(camera)
    extraction = 'do_system(f"ffmpeg -y -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png")'
    source = source.replace(extraction, f'print("Reusing verified frames", cam_name) if cam_name in {complete_cameras!r} else {extraction}')
    # Retain first-frame COLMAP source inputs and diagnostics for reproducibility.
    source = source.replace('shutil.rmtree(colmap_workspace)', "print('[INFO] Keeping COLMAP workspace for audit')")
    generated = ROOT / 'deployment/n3v2blender_windows.py'
    generated.write_text(source)
    subprocess.run([sys.executable, str(generated), scene.as_posix()], check=True, cwd=ROOT)
    report = {'image_scale_factor': 0.5, 'expected_render_width': 1352, 'expected_render_height': 1014,
              'max_timestamp_exclusive_seconds': 10.0, 'test_camera': 'cam00'}
    for split in ('train', 'test'):
        metadata = json.loads((scene / f'transforms_{split}.json').read_text())
        frames = [f for f in metadata['frames'] if f['time'] < 10.0]
        cameras = sorted({Path(f['file_path']).name[:5] for f in frames})
        if split == 'test' and cameras != ['cam00']:
            raise RuntimeError(f'Unexpected test cameras: {cameras}')
        report[split] = {'views': len(frames), 'cameras': cameras,
                         'timestamps_per_camera': {c: sum(Path(f['file_path']).name.startswith(c) for f in frames) for c in cameras}}
    (ROOT / 'deployment/dataset_settings.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
