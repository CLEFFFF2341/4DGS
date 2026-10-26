"""Download the authors' real CDAVIS archive and verify its pinned digest."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

URL = 'https://download.ifi.uzh.ch/rpg/web/data/cvpr24_evdeblurnerf/evdeblurnerf_cdavis.zip'
SHA256 = '1117dd16caef0cc4e05db830119edf00860177bc7415f6778cc2bfb65e7c200a'


def main():
    root = Path('dataset/evdeblurnerf_real')
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'evdeblurnerf_cdavis.zip'
    if not archive.exists():
        # The local proxy returned CONNECT 403; the public host works directly.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        partial = archive.with_suffix('.zip.partial')
        with opener.open(URL, timeout=60) as response, partial.open('wb') as f:
            while chunk := response.read(4 * 1024 * 1024):
                f.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('Archive checksum mismatch; partial download retained')
        partial.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('Existing archive checksum mismatch')
    destination = root / 'ev-deblurnerf_cdavis'
    if not destination.exists():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
                    raise ValueError('Unsafe archive member')
            bad = z.testzip()
            if bad: raise RuntimeError(f'ZIP CRC failed: {bad}')
            z.extractall(root)
    manifest = {'url': URL, 'sha256': SHA256, 'bytes': archive.stat().st_size,
                'publication': 'https://rpg.ifi.uzh.ch/docs/CVPR24_Cannici.pdf',
                'repository': 'https://github.com/uzh-rpg/EvDeblurNeRF',
                'kind': 'real captured CDAVIS, not Blender synthetic'}
    (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__': main()
