from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .config import sha256_json


@torch.no_grad()
def build_k0(adapter, times: list[int], output_root: Path) -> dict:
    specification = {
        "schema": "k0-v1",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "times": times,
        "interp_type": adapter.args.interp_type,
        "rot_interp_type": adapter.args.rot_interp_type,
    }
    key = sha256_json(specification)
    directory = output_root / key
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "k0.pt"
    if final.exists():
        return {"key": key, "path": str(final), "reused": True}
    started = time.perf_counter()
    payload = {
        "specification": specification,
        "static_rows": torch.from_numpy(adapter.static_rows.copy()),
        "dynamic_rows": torch.from_numpy(adapter.dynamic_rows.copy()),
        "positions": torch.stack([adapter.model.get_xyz_at_t(t, training=False).cpu() for t in times]),
        "rotations": torch.stack([adapter.model.get_rotation_at_t(t).cpu() for t in times]),
        "opacities": torch.stack([adapter.model.get_opacity_at_t(t, training=False).cpu() for t in times]),
        "scaling": adapter.model.get_scaling().cpu(),
        "features": adapter.model.get_features().cpu(),
    }
    temporary = directory / "k0.pt.partial"
    torch.save(payload, temporary)
    temporary.replace(final)
    metadata = {
        "key": key,
        "path": str(final),
        "bytes": final.stat().st_size,
        "seconds": time.perf_counter() - started,
        "reused": False,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
