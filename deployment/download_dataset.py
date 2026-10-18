"""Download the official coffee_martini archive with resumable HTTP ranges."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip'
SIZE = 1186324684
EXPECTED_MD5 = 'c12a2ae2fd216b2764e52f814474ea5c'
PARTS = ROOT / 'downloads' / 'coffee_martini.parts'

def fetch(item):
    index, start, end = item
    path = PARTS / f'{index:03d}.part'
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(f'Range request returned {response.status}')
                content_range = response.headers.get('Content-Range', '')
                if not content_range.startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError(f'Unexpected range: {content_range}')
                with path.open('wb') as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if path.stat().st_size != expected:
                raise RuntimeError(f'Incomplete part {index}')
            print(f'part {index + 1}/32 complete', flush=True)
            return
        except Exception as error:
            print(f'part {index}: attempt {attempt + 1}: {error}', flush=True)
            if attempt == 4:
                raise
            time.sleep(2)

def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    width = (SIZE + 31) // 32
    jobs = [(i, i * width, min(SIZE - 1, (i + 1) * width - 1)) for i in range(32)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, jobs))
    target = ROOT / 'downloads' / 'coffee_martini.verified.zip'
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with target.open('wb') as output:
        for i, _, _ in jobs:
            with (PARTS / f'{i:03d}.part').open('rb') as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    md5.update(block)
                    sha256.update(block)
    if md5.hexdigest() != EXPECTED_MD5:
        raise RuntimeError(f'Archive MD5 mismatch: {md5.hexdigest()}')
    result = dict(url=URL, size=SIZE, md5=md5.hexdigest(), sha256=sha256.hexdigest())
    (ROOT / 'deployment' / 'dataset_archive.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
