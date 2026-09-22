from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .adapter import DYNAMIC_PARAMETERS, STATIC_PARAMETERS, load_bundle
from .evaluate import render_one, select_camera


@torch.no_grad()
def run_adapter_checks(adapter, scene, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    deleted_static = np.sort(rng.choice(adapter.static_count, size=10, replace=False))
    deleted_dynamic = np.sort(rng.choice(adapter.dynamic_count, size=10, replace=False))
    kept_static = np.setdiff1d(np.arange(adapter.static_count), deleted_static, assume_unique=True)
    kept_dynamic = np.setdiff1d(np.arange(adapter.dynamic_count), deleted_dynamic, assume_unique=True)
    subset = adapter.gather(kept_static, kept_dynamic)
    tensor_checks = {}
    for name in STATIC_PARAMETERS:
        expected = getattr(adapter.model, name).index_select(0, torch.as_tensor(kept_static, device="cuda"))
        tensor_checks[name] = bool(torch.equal(expected, getattr(subset.model, name)))
    for name in DYNAMIC_PARAMETERS:
        expected = getattr(adapter.model, name).index_select(0, torch.as_tensor(kept_dynamic, device="cuda"))
        tensor_checks[name] = bool(torch.equal(expected, getattr(subset.model, name)))

    camera = select_camera(scene, "cam03", 149)
    deterministic_a = render_one(adapter, camera)
    deterministic_b = render_one(adapter, camera)
    deterministic_max_abs = float((deterministic_a - deterministic_b).abs().max())

    bundle_dir = output_root / "reference_bundle"
    if not bundle_dir.exists():
        bundle_metadata = adapter.save_bundle(bundle_dir)
    else:
        bundle_metadata = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    reloaded = load_bundle(bundle_dir, adapter.args)
    roundtrip = {}
    for timestamp in (0.0, 0.5, 149.0, 299.0):
        temporal_camera = copy.copy(select_camera(scene, "cam03", int(timestamp)))
        temporal_camera.timestamp = timestamp
        original = render_one(adapter, temporal_camera)
        restored = render_one(reloaded, temporal_camera)
        roundtrip[str(timestamp)] = float((original - restored).abs().max())

    dynamic_only = adapter.gather([], range(adapter.dynamic_count))
    static_only = adapter.gather(range(adapter.static_count), [])
    empty_kind = {
        "dynamic_only_shape": list(render_one(dynamic_only, camera).shape),
        "static_only_shape": list(render_one(static_only, camera).shape),
    }
    result = {
        "status": "COMPLETED",
        "seed": 0,
        "deleted_static_rows": deleted_static.tolist(),
        "deleted_dynamic_rows": deleted_dynamic.tolist(),
        "true_means_delete": bool(not np.isin(deleted_static, subset.static_rows).any() and not np.isin(deleted_dynamic, subset.dynamic_rows).any()),
        "tensor_gather_checks": tensor_checks,
        "all_tensor_gathers_passed": all(tensor_checks.values()),
        "deterministic_max_abs": deterministic_max_abs,
        "bundle": bundle_metadata,
        "roundtrip_max_abs": roundtrip,
        "roundtrip_passed": all(value <= 1e-6 for value in roundtrip.values()),
        "empty_kind": empty_kind,
    }
    (output_root / "adapter_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
