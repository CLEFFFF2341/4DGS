from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapter import load_bundle, load_reference
from .config import ROOT, git_provenance, sha256_file
from .evaluate import (
    benchmark_latency,
    evaluate_against_reference,
    render_one,
    save_comparison_visualizations,
    select_camera,
    smoke_render,
)
from .methods.p13 import apply_merges, build_pair_plan, moment_toy
from .runner import gpu_lock


OUTPUT_ROOT = ROOT / "runs" / "research" / "P13" / "cut_roasted_beef" / "b0.5"
P02_ROOT = ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def find_cache(filename: str, predicate) -> Path:
    matches = []
    for path in (ROOT / "research_cache" / "cut_roasted_beef").glob(f"*/{filename}"):
        value = torch.load(path, map_location="cpu")
        if predicate(value.get("specification", {})):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} cache, got {matches}")
    return matches[0]


def reload_check(expected, restored, scene) -> dict[str, Any]:
    camera = select_camera(scene, "cam01", 149)
    difference = (render_one(expected, camera) - render_one(restored, camera)).abs()
    result = {
        "static_rows_equal": bool(np.array_equal(expected.static_rows, restored.static_rows)),
        "dynamic_rows_equal": bool(np.array_equal(expected.dynamic_rows, restored.dynamic_rows)),
        "render_max_abs": float(difference.max()),
        "render_mean_abs": float(difference.mean()),
    }
    result["passed"] = bool(
        result["static_rows_equal"] and result["dynamic_rows_equal"] and result["render_max_abs"] <= 1e-6
    )
    return result


def disjoint(pairs: np.ndarray) -> bool:
    if not pairs.size:
        return True
    endpoints = pairs[:, :2].astype(np.int64).reshape(-1)
    return np.unique(endpoints).size == endpoints.size


def probe(reference, scene, plan, k0) -> dict[str, Any]:
    control = reference.gather(plan.static_indices, plan.dynamic_indices)
    merged, applied = apply_merges(reference, plan, k0, max_pairs=16)
    rows = []
    for timestamp in (0, 149):
        camera = select_camera(scene, "cam01", timestamp)
        control_image = render_one(control, camera)
        merged_image = render_one(merged, camera)
        difference = (merged_image - control_image).abs()
        rows.append(
            {
                "camera": "cam01",
                "timestamp": timestamp,
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "merged_finite": bool(torch.isfinite(merged_image).all()),
            }
        )
    result = {
        "status": "COMPLETED",
        "requested_total_pairs": 32,
        "applied": applied,
        "frames": rows,
        "finite": all(row["merged_finite"] for row in rows),
    }
    result["passed"] = result["finite"] and sum(applied.values()) == 32
    return result


def correctness(reference, plan, merged, control, scene) -> dict[str, Any]:
    static_pairs = plan.static_pairs
    dynamic_pairs = plan.dynamic_pairs
    repeats = [
        build_pair_plan(reference, correctness.k0, correctness.k4, correctness.scores)
        for _ in range(2)
    ]
    camera = select_camera(scene, "cam01", 149)
    identity = reference.gather(range(reference.static_count), range(reference.dynamic_count))
    identity_error = float((render_one(reference, camera) - render_one(identity, camera)).abs().max())
    static_selected = set(plan.static_indices.tolist())
    dynamic_selected = set(plan.dynamic_indices.tolist())
    static_parents = np.asarray(
        [int(pair[0]) for pair in static_pairs if int(pair[0]) in static_selected],
        dtype=np.int64,
    )
    dynamic_parents = np.asarray(
        [int(pair[0]) for pair in dynamic_pairs if int(pair[0]) in dynamic_selected],
        dtype=np.int64,
    )
    static_rows = torch.from_numpy(np.searchsorted(plan.static_indices, static_parents)).cuda()
    dynamic_rows = torch.from_numpy(np.searchsorted(plan.dynamic_indices, dynamic_parents)).cuda()
    quaternion_norm_static = torch.linalg.vector_norm(merged.model._rotation.index_select(0, static_rows), dim=1)
    quaternion_norm_dynamic = torch.linalg.vector_norm(
        merged.model._rotation_motion.index_select(0, dynamic_rows), dim=2
    )
    checks = {
        "moment_toy": moment_toy(),
        "static_pairs_disjoint": disjoint(static_pairs),
        "dynamic_pairs_disjoint": disjoint(dynamic_pairs),
        "parent_ids_unique": bool(
            np.unique(static_pairs[:, 0].astype(np.int64)).size == static_pairs.shape[0]
            and np.unique(dynamic_pairs[:, 0].astype(np.int64)).size == dynamic_pairs.shape[0]
        ),
        "total_k_exact": merged.total_count == control.total_count == 124447,
        "r0_r1_ids_equal": bool(
            np.array_equal(merged.static_rows, control.static_rows)
            and np.array_equal(merged.dynamic_rows, control.dynamic_rows)
        ),
        "full_retention_identity_max_abs": identity_error,
        "full_retention_identity": identity_error <= 1e-6,
        "opacity_finite": bool(
            torch.isfinite(merged.model._opacity).all() and torch.isfinite(merged.model._opacity_motion).all()
        ),
        "scale_finite": bool(
            torch.isfinite(merged.model._scaling).all() and torch.isfinite(merged.model._scaling_motion).all()
        ),
        "static_quaternion_max_norm_error": float((quaternion_norm_static - 1).abs().max()),
        "dynamic_quaternion_max_norm_error": float((quaternion_norm_dynamic - 1).abs().max()),
        "quaternions_normalized": bool(
            float((quaternion_norm_static - 1).abs().max()) <= 1e-5
            and float((quaternion_norm_dynamic - 1).abs().max()) <= 1e-5
        ),
        "deterministic_seed_policy_0_1_2": all(
            np.array_equal(plan.static_indices, repeated.static_indices)
            and np.array_equal(plan.dynamic_indices, repeated.dynamic_indices)
            and np.array_equal(plan.static_pairs, repeated.static_pairs)
            and np.array_equal(plan.dynamic_pairs, repeated.dynamic_pairs)
            for repeated in repeats
        ),
    }
    checks["passed"] = bool(
        checks["moment_toy"]["passed"]
        and all(value for key, value in checks.items() if isinstance(value, bool))
    )
    return checks


def run_rule(reference, scene, adapter, plan, splits, samples, config, rule: str, cache_paths) -> dict[str, Any]:
    output = OUTPUT_ROOT / rule
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    np.savez_compressed(
        output / "kept_ids.npz",
        static_indices=plan.static_indices,
        dynamic_indices=plan.dynamic_indices,
        static_rows=reference.static_rows[plan.static_indices],
        dynamic_rows=reference.dynamic_rows[plan.dynamic_indices],
    )
    bundle_dir = output / "bundle"
    if not bundle_dir.exists():
        bundle = adapter.save_bundle(bundle_dir)
    else:
        bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    roundtrip = reload_check(adapter, load_bundle(bundle_dir, reference.args), scene)
    if not roundtrip["passed"]:
        raise RuntimeError(roundtrip)
    smoke = smoke_render(adapter, scene, [splits["development"][0]], [0, 149], output / "smoke")
    summary = evaluate_against_reference(
        adapter, reference, scene, splits["development"], samples["T24"], output / "evaluation"
    )
    inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
    latency = benchmark_latency(adapter, scene, inputs, output / "latency.json")
    save_comparison_visualizations(
        adapter,
        reference,
        scene,
        splits["development"],
        samples["fixed_visualization_times"],
        output / "visualizations",
    )
    total_seconds = time.perf_counter() - started
    resources = {
        "actual_total_points": adapter.total_count,
        "static_points": adapter.static_count,
        "dynamic_points": adapter.dynamic_count,
        "bundle_bytes": directory_bytes(bundle_dir),
        "smoke": smoke,
        "latency": latency,
    }
    dump(output / "resolved_config.json", {**config, "rule": rule})
    dump(output / "bundle_reload_check.json", roundtrip)
    dump(output / "timing.json", {"total_seconds": total_seconds})
    dump(output / "resources.json", resources)
    dump(
        output / "provenance.json",
        {
            "git": git_provenance(),
            "checkpoint_sha256": reference.checkpoint_sha256,
            "cache_sha256": {name: sha256_file(path) for name, path in cache_paths.items()},
            "config_sha256": sha256_file(ROOT / "configs" / "research" / "p13.json"),
        },
    )
    (output / "commands.txt").write_text(f"python -m research_impl.p13_runner  # {rule}\n", encoding="utf-8")
    result = {
        "status": "COMPLETED",
        "rule": rule,
        "summary": summary,
        "resources": resources,
        "bundle": bundle,
    }
    dump(output / "result.json", result)
    dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs" / "research" / "p13.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    k0_path = find_cache("k0.pt", lambda spec: spec.get("times") == samples["T24"])
    k4_path = find_cache(
        "k4.pt",
        lambda spec: spec.get("times") == samples["T24"]
        and spec.get("neighbors_excluding_self") == config["neighbors"]
        and spec.get("position_reference_time") == config["position_reference_time"],
    )
    k0 = torch.load(k0_path, map_location="cpu")
    k4 = torch.load(k4_path, map_location="cpu")
    scores = np.asarray(torch.load(P02_ROOT / "selection.pt", map_location="cpu")["scores"], dtype=np.float64)
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        selection_started = time.perf_counter()
        plan = build_pair_plan(reference, k0, k4, scores)
        selection_seconds = time.perf_counter() - selection_started
        np.savez_compressed(
            OUTPUT_ROOT / "pair_plan.npz",
            static_indices=plan.static_indices,
            dynamic_indices=plan.dynamic_indices,
            static_pairs=plan.static_pairs,
            dynamic_pairs=plan.dynamic_pairs,
            static_weights=plan.static_weights,
            dynamic_weights=plan.dynamic_weights,
        )
        dump(OUTPUT_ROOT / "pair_diagnostics.json", plan.diagnostics)
        probe_result = probe(reference, scene, plan, k0)
        dump(OUTPUT_ROOT / "probe.json", probe_result)
        if not probe_result["passed"]:
            raise RuntimeError(probe_result)
        transform_started = time.perf_counter()
        merged, applied = apply_merges(reference, plan, k0)
        transform_seconds = time.perf_counter() - transform_started
        control = reference.gather(plan.static_indices, plan.dynamic_indices)
        correctness.k0 = k0
        correctness.k4 = k4
        correctness.scores = scores
        checks = correctness(reference, plan, merged, control, scene)
        checks["probe"] = probe_result
        dump(OUTPUT_ROOT / "correctness_checks.json", checks)
        if not checks["passed"]:
            raise RuntimeError(checks)
        cache_paths = {"K0": k0_path, "K2_scores": P02_ROOT / "selection.pt", "K4": k4_path}
        r0 = run_rule(reference, scene, merged, plan, splits, samples, config, "R0", cache_paths)
        r1 = run_rule(reference, scene, control, plan, splits, samples, config, "R1", cache_paths)
    aggregate = {
        "status": "COMPLETED",
        "seed_policy": "deterministic IDs verified for seeds 0,1,2; one quality evaluation per rule",
        "pair_diagnostics": plan.diagnostics,
        "applied_merges": applied,
        "selection_seconds": selection_seconds,
        "transform_seconds": transform_seconds,
        "R0": r0,
        "R1": r1,
        "delta_psnr_r0_minus_r1": r0["summary"]["mean_psnr"] - r1["summary"]["mean_psnr"],
        "delta_lpips_r0_minus_r1": r0["summary"]["mean_lpips_alex"] - r1["summary"]["mean_lpips_alex"],
    }
    dump(OUTPUT_ROOT / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
