"""Fetch a pinned GitHub release tool with parallel byte ranges and SHA256 verification."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import download_dataset as downloader

ROOT = Path(__file__).resolve().parents[1]
release, asset_name, target_name = sys.argv[1:]
metadata = json.loads((ROOT / 'deployment' / release).read_text())
asset = next(a for a in metadata['assets'] if a['name'] == asset_name)
downloader.URL = asset['browser_download_url']
downloader.SIZE = asset['size']
downloader.PARTS = ROOT / 'downloads' / (target_name + '.parts')
downloader.PARTS.mkdir(parents=True, exist_ok=True)
width = (asset['size'] + 31) // 32
jobs = [(i, i * width, min(asset['size'] - 1, (i + 1) * width - 1)) for i in range(32)]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(downloader.fetch, jobs))
target = ROOT / 'downloads' / (target_name + '.verified.zip')
sha256 = hashlib.sha256()
with target.open('wb') as output:
    for i, _, _ in jobs:
        with (downloader.PARTS / f'{i:03d}.part').open('rb') as source:
            while block := source.read(1024 * 1024):
                output.write(block)
                sha256.update(block)
expected = asset.get('digest')
if not expected or expected != 'sha256:' + sha256.hexdigest():
    raise RuntimeError(f'Tool digest mismatch or missing digest: {expected}, {sha256.hexdigest()}')
with zipfile.ZipFile(target) as z:
    if z.testzip() is not None:
        raise RuntimeError('Tool ZIP CRC failed')
    if any('..' in Path(n).parts or Path(n).is_absolute() for n in z.namelist()):
        raise RuntimeError('Unsafe ZIP member')
    z.extractall(ROOT / 'tools' / target_name)
(ROOT / 'deployment' / (target_name + '_archive.json')).write_text(json.dumps({
    'url': downloader.URL, 'size': asset['size'], 'sha256': sha256.hexdigest()
}, indent=2))
print(f'{target_name} verified and extracted', flush=True)
