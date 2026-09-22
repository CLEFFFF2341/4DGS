from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace

from gaussian_renderer import render

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


@torch.no_grad()
def probe_k2(adapter, scene, camera_name: str, times: list[int], resolution: list[int], output: Path) -> dict:
    width, height = map(int, resolution)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    deletion_pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        collect_stats=True,
        collect_deletions=True,
    )
    plain_pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        collect_stats=False,
        collect_deletions=False,
    )
    frame_results = []
    aggregate_score = torch.zeros(adapter.total_count, dtype=torch.float64, device="cuda")
    frames = {}
    for timestamp in times:
        camera = copy.copy(select_camera(scene, camera_name, timestamp))
        camera.image_width = width
        camera.image_height = height
        statistics = render(camera, adapter.model, deletion_pipe, background, timestamp=timestamp, near=adapter.args.near, far=adapter.args.far)
        replay_difference = (statistics["render"] - statistics["replay_color"]).abs()
        frame_results.append({
            "timestamp": timestamp,
            "replay_max_abs": float(replay_difference.max()),
            "replay_mean_abs": float(replay_difference.mean()),
        })
        aggregate_score += statistics["contrib_sum"].double()
        frames[timestamp] = (camera, statistics)

    ns = adapter.static_count
    groups = {"static": aggregate_score[:ns], "dynamic": aggregate_score[ns:]}
    selected = []
    for kind, score in groups.items():
        nonzero = torch.nonzero(score > 0, as_tuple=False).flatten().cpu().numpy()
        values = score[torch.as_tensor(nonzero, device="cuda")].cpu().numpy()
        order = np.lexsort((nonzero, values))
        low_positions = np.linspace(0, max(len(order) // 5 - 1, 0), 8, dtype=int)
        high_start = max(4 * len(order) // 5, 0)
        high_positions = np.linspace(high_start, len(order) - 1, 8, dtype=int)
        for band, positions in (("low", low_positions), ("high", high_positions)):
            for position in positions:
                row = int(nonzero[order[position]])
                selected.append({"kind": kind, "row": row, "band": band, "global_index": row if kind == "static" else ns + row})

    comparisons = []
    pixel_count = width * height
    for timestamp in times:
        camera, statistics = frames[timestamp]
        full = statistics["render"]
        for item in selected:
            parameter = adapter.model._opacity if item["kind"] == "static" else adapter.model._opacity_motion
            row = item["row"]
            original = parameter[row].detach().clone()
            parameter[row].fill_(-100.0)
            deleted = render(camera, adapter.model, plain_pipe, background, timestamp=timestamp, near=adapter.args.near, far=adapter.args.far)["render"]
            parameter[row].copy_(original)
            brute = float(torch.mean((full - deleted) ** 2))
            proxy = float(statistics["deletion_sq_sum"][item["global_index"]] / pixel_count)
            error = abs(proxy - brute)
            relative = error / brute if brute >= 1e-10 else None
            passed = error <= 1e-8 if brute < 1e-10 else relative <= 0.05
            comparisons.append({
                **item,
                "timestamp": timestamp,
                "brute_mse": brute,
                "proxy_mse": proxy,
                "absolute_error": error,
                "relative_error": relative,
                "passed": passed,
            })
    finite_relative = [row["relative_error"] for row in comparisons if row["relative_error"] is not None]
    result = {
        "status": "COMPLETED",
        "camera": camera_name,
        "times": times,
        "resolution": resolution,
        "selection": selected,
        "frames": frame_results,
        "comparisons": comparisons,
        "passed_count": sum(row["passed"] for row in comparisons),
        "comparison_count": len(comparisons),
        "median_relative_error": float(np.median(finite_relative)) if finite_relative else None,
        "max_relative_error": float(np.max(finite_relative)) if finite_relative else None,
        "proxy_valid": all(row["passed"] for row in comparisons) and max(row["replay_max_abs"] for row in frame_results) <= 1e-4,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
