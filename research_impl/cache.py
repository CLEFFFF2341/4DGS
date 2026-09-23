from __future__ import annotations

import json
import copy
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from gaussian_renderer import render

from .config import sha256_file, sha256_json
from .evaluate import pipeline, select_camera
from utils.sh_utils import SH2RGB


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


@torch.no_grad()
def build_k2(adapter, scene, camera_names: list[str], times: list[int], resolution: list[int], output_root: Path) -> dict:
    """Build exact single-deletion MSE using the renderer's early-stop control flow."""
    import diff_gaussian_rasterization_df._C as raster_extension

    extension_path = Path(raster_extension.__file__)
    specification = {
        "schema": "k2-exact-early-stop-deletion-v2",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "camera_names": camera_names,
        "times": times,
        "resolution": resolution,
        "background": [0.0, 0.0, 0.0],
        "alpha_cap": 0.99,
        "alpha_floor": 1.0 / 255.0,
        "early_stop_T": 1e-4,
        "counterfactual": "remove matching Gaussian ID and replay exact early-stop semantics, including the stopping gate",
        "extension_sha256": sha256_file(extension_path),
    }
    key = sha256_json(specification)
    directory = output_root / key
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "k2.pt"
    if final.exists():
        return {"key": key, "path": str(final), "reused": True}
    n = adapter.total_count
    width, height = map(int, resolution)
    e_it = np.zeros((len(times), n), dtype=np.float64)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    deletion_pipe = pipeline()
    deletion_pipe.collect_stats = True
    deletion_pipe.collect_deletions = True
    frame_timings = []
    replay_max_abs = 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    total_frames = len(times) * len(camera_names)
    completed = 0
    for time_index, timestamp in enumerate(times):
        for name in camera_names:
            camera = copy.copy(select_camera(scene, name, timestamp))
            camera.image_width = width
            camera.image_height = height
            frame_started = time.perf_counter()
            stats = render(
                camera,
                adapter.model,
                deletion_pipe,
                background,
                timestamp=timestamp,
                near=adapter.args.near,
                far=adapter.args.far,
            )
            seconds = time.perf_counter() - frame_started
            frame_replay = float((stats["render"] - stats["replay_color"]).abs().max())
            replay_max_abs = max(replay_max_abs, frame_replay)
            e_it[time_index] += stats["deletion_sq_sum"].cpu().numpy().astype(np.float64) / (width * height * len(camera_names))
            completed += 1
            frame_timings.append({"camera": name, "timestamp": timestamp, "seconds": seconds, "replay_max_abs": frame_replay})
            print(f"K2 exact deletion {completed}/{total_frames}: {name} t={timestamp}, {seconds:.3f}s", flush=True)
    payload = {
        "specification": specification,
        "static_rows": torch.from_numpy(adapter.static_rows.copy()),
        "dynamic_rows": torch.from_numpy(adapter.dynamic_rows.copy()),
        "e_it": torch.from_numpy(e_it),
        "mean_e_i": torch.from_numpy(e_it.mean(axis=0)),
        "frame_timings": frame_timings,
        "replay_max_abs": replay_max_abs,
    }
    temporary = directory / "k2.pt.partial"
    torch.save(payload, temporary)
    temporary.replace(final)
    metadata = {
        "key": key,
        "path": str(final),
        "bytes": final.stat().st_size,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nonzero_points": int((e_it.sum(axis=0) > 0).sum()),
        "replay_max_abs": replay_max_abs,
        "mean_frame_seconds": float(np.mean([row["seconds"] for row in frame_timings])),
        "reused": False,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@torch.no_grad()
def build_k4(adapter, times: list[int], output_root: Path) -> dict:
    """Build the P05 trajectory-feature and same-kind 16-NN cache."""
    specification = {
        "schema": "k4-trajectory-knn-v1",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "times": times,
        "neighbors_excluding_self": 16,
        "position_reference_time": 149,
        "standardization": "global median/IQR; IQR<1e-6->1; clip[-10,10]",
        "similarity": "exp(-squared_distance/21)",
    }
    key = sha256_json(specification)
    directory = output_root / key
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "k4.pt"
    if final.exists():
        return {"key": key, "path": str(final), "reused": True}
    started = time.perf_counter()
    positions = torch.stack([adapter.model.get_xyz_at_t(timestamp, training=False).cpu() for timestamp in times]).permute(1, 0, 2).double()
    position_149 = adapter.model.get_xyz_at_t(149, training=False).cpu().double()
    bounds = position_149.amax(dim=0) - position_149.amin(dim=0)
    length = float(torch.linalg.vector_norm(bounds))
    if length <= 1e-12:
        length = 1.0
    dc = SH2RGB(adapter.model.get_features()[:, 0, :].detach().cpu().double())
    logscale = torch.log(adapter.model.get_scaling().detach().cpu().double().clamp_min(1e-30))
    raw = torch.cat((positions.reshape(adapter.total_count, -1) / length, dc, logscale), dim=1)
    median = torch.quantile(raw, 0.5, dim=0)
    q25 = torch.quantile(raw, 0.25, dim=0)
    q75 = torch.quantile(raw, 0.75, dim=0)
    iqr = q75 - q25
    iqr = torch.where(iqr < 1e-6, torch.ones_like(iqr), iqr)
    features = ((raw - median) / iqr).clamp(-10, 10).float().numpy()
    xyz = position_149.numpy()
    neighbor_parts = []
    similarity_parts = []
    brute_checks = []
    offset = 0
    for kind, count in (("static", adapter.static_count), ("dynamic", adapter.dynamic_count)):
        local_xyz = xyz[offset : offset + count]
        local_features = features[offset : offset + count]
        tree = cKDTree(local_xyz)
        query_k = min(33, count)
        distances, candidates = tree.query(local_xyz, k=query_k, workers=-1)
        if query_k == 1:
            distances = distances[:, None]
            candidates = candidates[:, None]
        keep_k = min(17, count)
        neighbors = np.empty((count, keep_k), dtype=np.int32)
        for row in range(count):
            order = np.lexsort((candidates[row], distances[row]))
            neighbors[row] = candidates[row, order[:keep_k]]
            if row not in neighbors[row]:
                neighbors[row, -1] = row
        delta = local_features[:, None, :] - local_features[neighbors]
        similarity = np.exp(-np.sum(delta * delta, axis=2) / 21.0).astype(np.float32)
        self_column = np.argmax(neighbors == np.arange(count, dtype=np.int32)[:, None], axis=1)
        similarity[np.arange(count), self_column] = 1.0
        neighbor_parts.append(torch.from_numpy(neighbors))
        similarity_parts.append(torch.from_numpy(similarity))

        sample = np.linspace(0, count - 1, min(64, count), dtype=np.int64)
        subset_xyz = local_xyz[sample]
        subset_tree = cKDTree(subset_xyz)
        kd_distance, kd_index = subset_tree.query(subset_xyz, k=min(17, len(sample)))
        brute_distance = np.linalg.norm(subset_xyz[:, None, :] - subset_xyz[None, :, :], axis=2)
        brute_index = np.argsort(brute_distance, axis=1, kind="stable")[:, : min(17, len(sample))]
        brute_checks.append({
            "kind": kind,
            "subset_size": int(len(sample)),
            "neighbor_sets_equal": bool(all(set(np.atleast_1d(kd_index[row]).tolist()) == set(brute_index[row].tolist()) for row in range(len(sample)))),
            "max_distance_error": float(np.max(np.abs(np.take_along_axis(brute_distance, np.atleast_2d(kd_index), axis=1) - np.atleast_2d(kd_distance)))) if len(sample) else 0.0,
        })
        offset += count
    payload = {
        "specification": specification,
        "static_rows": torch.from_numpy(adapter.static_rows.copy()),
        "dynamic_rows": torch.from_numpy(adapter.dynamic_rows.copy()),
        "features": torch.from_numpy(features),
        "static_neighbors": neighbor_parts[0],
        "dynamic_neighbors": neighbor_parts[1],
        "static_similarity": similarity_parts[0],
        "dynamic_similarity": similarity_parts[1],
        "length_scale": length,
        "median": median,
        "iqr": iqr,
        "brute_checks": brute_checks,
    }
    temporary = directory / "k4.pt.partial"
    torch.save(payload, temporary)
    temporary.replace(final)
    metadata = {
        "key": key,
        "path": str(final),
        "bytes": final.stat().st_size,
        "seconds": time.perf_counter() - started,
        "feature_shape": list(features.shape),
        "brute_checks": brute_checks,
        "reused": False,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
