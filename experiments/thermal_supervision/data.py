"""Explicit calibrated manifest; no filename pairing or guessed camera poses."""
import json
from pathlib import Path
import numpy as np
import torch
from .core import Frame
from .runtime import Camera, ExposureSample, Observation


def load_manifest(path, device='cuda'):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema') != 1 or document.get('units') != 'linear_sensor':
        raise ValueError('schema 1 with explicitly calibrated linear_sensor units required')
    if document.get('static_scene') is not True:
        raise ValueError('CLI currently supports audited static scenes only')
    rows = document['observations']
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate observation IDs')
    transactions = document['transactions']
    if not transactions:
        raise ValueError('at least one transaction required')
    selected = set()
    for item in transactions:
        selected.update([item['source'], *item['witnesses'], *item['guards']])
    lookup = {row['id']: row for row in rows}
    # Audit every selected split BEFORE opening any sensor arrays.
    for key in selected:
        if lookup[key]['split'] != 'train' or lookup[key].get('calibrated') is not True:
            raise ValueError('only explicitly calibrated training observations can drive updates')
    def array(name):
        p = (path.parent/name).resolve()
        if not p.is_relative_to(path.parent) or p.suffix != '.npy':
            raise ValueError('arrays must be local .npy files inside manifest directory')
        value = np.load(p, allow_pickle=False)
        if value.dtype not in (np.float32, np.float64):
            raise ValueError('sensor arrays require floating linear values')
        return torch.from_numpy(value).float().to(device)
    observations = {}
    for key in sorted(selected):
        row = lookup[key]
        samples = []
        for sample in row['exposure']:
            camera = dict(sample['camera'])
            camera['w2c'] = torch.tensor(camera['w2c'], dtype=torch.float32, device=device)
            samples.append(ExposureSample(Camera(**camera), sample['model_time'], sample['weight']))
        obs = Observation(Frame(key, row['viewpoint_id'], row['timestamp_seconds'], row['split'], True),
            row['sensor'], tuple(samples), array(row['target']), array(row['weight']),
            torch.tensor(row['noise'], dtype=torch.float32, device=device), row['gain'], row['bias'])
        obs.validate()
        observations[key] = obs
    return document, [(observations[t['source']], [observations[k] for k in t['witnesses']],
                      [observations[k] for k in t['guards']]) for t in transactions]
