from __future__ import annotations

import json
import copy
import time
from pathlib import Path

import numpy as np
import torch

from gaussian_renderer import render

from .config import sha256_file, sha256_json
from .evaluate import pipeline, select_camera


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


@torch.no_grad()
def build_k1(adapter, scene, camera_names: list[str], times: list[int], resolution: list[int], output_root: Path) -> dict:
    import diff_gaussian_rasterization_df._C as raster_extension

    extension_path = Path(raster_extension.__file__)
    specification = {
        "schema": "k1-true-contribution-v1",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "camera_names": camera_names,
        "times": times,
        "resolution": resolution,
        "background": [0.0, 0.0, 0.0],
        "alpha_cap": 0.99,
        "alpha_floor": 1.0 / 255.0,
        "early_stop_T": 1e-4,
        "extension_sha256": sha256_file(extension_path),
    }
    key = sha256_json(specification)
    directory = output_root / key
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "k1.pt"
    if final.exists():
        return {"key": key, "path": str(final), "reused": True}
    n = adapter.total_count
    width, height = map(int, resolution)
    s_it = np.zeros((len(times), n), dtype=np.float64)
    hit_counts = np.zeros((len(times), n), dtype=np.int64)
    transmittance_sum = np.zeros((len(times), n), dtype=np.float64)
    tile_touches = np.zeros((len(times), n), dtype=np.int64)
    max_weight = np.zeros(n, dtype=np.float32)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    stats_pipe = pipeline()
    stats_pipe.collect_stats = True
    plain_pipe = pipeline()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    validation = None
    for time_index, timestamp in enumerate(times):
        for name in camera_names:
            camera = copy.copy(select_camera(scene, name, timestamp))
            camera.image_width = width
            camera.image_height = height
            stats = render(
                camera,
                adapter.model,
                stats_pipe,
                background,
                timestamp=timestamp,
                near=adapter.args.near,
                far=adapter.args.far,
            )
            s_it[time_index] += stats["contrib_sum"].cpu().numpy().astype(np.float64) / (width * height * len(camera_names))
            hit_counts[time_index] += stats["contrib_hit_count"].cpu().numpy().astype(np.int64)
            transmittance_sum[time_index] += stats["transmittance_sum"].cpu().numpy().astype(np.float64) / (width * height * len(camera_names))
            tile_touches[time_index] += stats["tiles_touched"].cpu().numpy().astype(np.int64)
            np.maximum(max_weight, stats["contrib_max"].cpu().numpy(), out=max_weight)
            if validation is None:
                plain = render(
                    camera,
                    adapter.model,
                    plain_pipe,
                    background,
                    timestamp=timestamp,
                    near=adapter.args.near,
                    far=adapter.args.far,
                )
                sum_contrib = float(stats["contrib_sum"].sum())
                sum_acc = float(stats["acc"].sum())
                validation = {
                    "stats_toggle_image_max_abs": float((plain["render"] - stats["render"]).abs().max()),
                    "sum_contrib": sum_contrib,
                    "sum_acc": sum_acc,
                    "relative_conservation_error": abs(sum_contrib - sum_acc) / max(abs(sum_acc), 1e-12),
                }
    opacities = torch.stack([adapter.model.get_opacity_at_t(timestamp, training=False).cpu() for timestamp in times])
    scaling = adapter.model.get_scaling().cpu()
    payload = {
        "specification": specification,
        "static_rows": torch.from_numpy(adapter.static_rows.copy()),
        "dynamic_rows": torch.from_numpy(adapter.dynamic_rows.copy()),
        "s_it": torch.from_numpy(s_it),
        "m_i": torch.from_numpy(max_weight),
        "hit_counts": torch.from_numpy(hit_counts),
        "transmittance_sum": torch.from_numpy(transmittance_sum),
        "tile_touches": torch.from_numpy(tile_touches),
        "visible_by_time": torch.from_numpy(s_it > 0),
        "opacity_by_time": opacities,
        "volume": scaling.prod(dim=1),
        "validation": validation,
    }
    temporary = directory / "k1.pt.partial"
    torch.save(payload, temporary)
    temporary.replace(final)
    metadata = {
        "key": key,
        "path": str(final),
        "bytes": final.stat().st_size,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nonzero_points": int((s_it.sum(axis=0) > 0).sum()),
        "validation": validation,
        "reused": False,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
