from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .adapter import load_bundle, load_reference
from .cache import build_k1
from .config import ROOT, git_provenance, sha256_file
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p01 import correctness_checks, select
from .runner import gpu_lock


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def safe_remove_probe(path: Path, probe_root: Path) -> None:
    resolved = path.resolve()
    root = probe_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove probe outside {root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def selection_for_count(adapter, cache, count: int):
    fraction = 1.0 if count == adapter.total_count else (count + 0.5) / adapter.total_count
    result = select(adapter, cache, "R4", fraction, 0)
    if result.diagnostics["actual_total"] != count:
        raise RuntimeError(f"Requested {count} points, selected {result.diagnostics['actual_total']}")
    return result


def bundle_probe(adapter, selection, directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=False)
    subset = adapter.gather(selection.static_indices, selection.dynamic_indices)
    subset.save_bundle(directory / "bundle")
    return directory_bytes(directory / "bundle")


def reload_check(expected, restored, scene, camera_name: str) -> dict:
    camera = select_camera(scene, camera_name, 149)
    difference = (render_one(expected, camera) - render_one(restored, camera)).abs()
    result = {
        "static_rows_equal": bool(np.array_equal(expected.static_rows, restored.static_rows)),
        "dynamic_rows_equal": bool(np.array_equal(expected.dynamic_rows, restored.dynamic_rows)),
        "render_max_abs": float(difference.max()),
        "render_mean_abs": float(difference.mean()),
    }
    result["passed"] = result["static_rows_equal"] and result["dynamic_rows_equal"] and result["render_max_abs"] <= 1e-6
    return result


def target_bytes(target: dict) -> int:
    resources = json.loads((ROOT / target["source"]).read_text(encoding="utf-8"))
    return int(resources["compact_bundle_bytes"])


def run_target(name: str, cap: int, max_probes: int, adapter, scene, cache, cache_meta, splits, samples, checks) -> dict:
    output = ROOT / "runs" / "research" / "resource_controls" / name / "P01_R4"
    if (output / "status.json").exists():
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETED":
            return json.loads((output / "result.json").read_text(encoding="utf-8"))
    probe_root = ROOT / "runs" / "research" / "resource_controls" / ".probes" / name
    probe_root.mkdir(parents=True, exist_ok=True)
    lower = 124447
    upper = adapter.total_count - 1
    best_probe = None
    best_selection = None
    best_bytes = None
    probes = []
    search_started = time.perf_counter()
    for probe_index in range(max_probes):
        if lower > upper:
            break
        count = (lower + upper + 1) // 2
        selection = selection_for_count(adapter, cache, count)
        probe_dir = probe_root / f"probe_{probe_index:02d}_{count}"
        if probe_dir.exists():
            safe_remove_probe(probe_dir, probe_root)
        size = bundle_probe(adapter, selection, probe_dir)
        passed = size <= cap
        probes.append(
            {
                "probe": probe_index,
                "points": count,
                "static_points": int(selection.static_indices.size),
                "dynamic_points": int(selection.dynamic_indices.size),
                "bundle_bytes": size,
                "cap_bytes": cap,
                "within_cap": passed,
            }
        )
        print(f"resource {name} probe {probe_index + 1}/{max_probes}: K={count}, bytes={size}, within={passed}", flush=True)
        if passed:
            if best_probe is not None and best_probe != probe_dir:
                safe_remove_probe(best_probe, probe_root)
            best_probe = probe_dir
            best_selection = selection
            best_bytes = size
            lower = count + 1
        else:
            safe_remove_probe(probe_dir, probe_root)
            upper = count - 1
    if best_probe is None or best_selection is None or best_bytes is None:
        raise RuntimeError(f"No P01-R4 bundle fit target {name} cap {cap}")
    search_seconds = time.perf_counter() - search_started

    output.mkdir(parents=True, exist_ok=True)
    final_bundle = output / "bundle"
    if final_bundle.exists():
        safe_remove_probe(final_bundle, output.parent)
    (best_probe / "bundle").replace(final_bundle)
    safe_remove_probe(best_probe, probe_root)
    subset = adapter.gather(best_selection.static_indices, best_selection.dynamic_indices)
    restored = load_bundle(final_bundle, adapter.args)
    roundtrip = reload_check(subset, restored, scene, splits["development"][0])
    if not roundtrip["passed"]:
        raise RuntimeError(roundtrip)
    np.savez_compressed(
        output / "kept_ids.npz",
        static_indices=best_selection.static_indices,
        dynamic_indices=best_selection.dynamic_indices,
        static_rows=adapter.static_rows[best_selection.static_indices],
        dynamic_rows=adapter.dynamic_rows[best_selection.dynamic_indices],
    )
    dump(output / "resolved_config.json", {"rule": "P01-R4", "seed": 0, "target_name": name, "cap_bytes": cap, "max_byte_probes": max_probes})
    (output / "commands.txt").write_text("python -m research_impl.resource_control_runner\n", encoding="utf-8")
    dump(output / "byte_probes.json", probes)
    smoke_started = time.perf_counter()
    smoke = smoke_render(subset, scene, [splits["development"][0]], [0, 149], output / "smoke")
    evaluation_started = time.perf_counter()
    summary = evaluate_against_reference(subset, adapter, scene, splits["development"], samples["T24"], output)
    evaluation_seconds = time.perf_counter() - evaluation_started
    inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
    latency = benchmark_latency(subset, scene, inputs, output / "latency.json")
    save_comparison_visualizations(subset, adapter, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
    timing = {
        "byte_search_seconds": search_seconds,
        "smoke_seconds": evaluation_started - smoke_started,
        "evaluation_seconds": evaluation_seconds,
    }
    resources = {
        "cap_bytes": cap,
        "bundle_bytes": best_bytes,
        "unused_bytes": cap - best_bytes,
        "actual_total_points": best_selection.diagnostics["actual_total"],
        "static_points": int(best_selection.static_indices.size),
        "dynamic_points": int(best_selection.dynamic_indices.size),
        "byte_probe_count": len(probes),
        "roundtrip": roundtrip,
        "smoke": smoke,
        "latency": latency,
    }
    dump(output / "resources.json", resources)
    dump(output / "timing.json", timing)
    dump(output / "correctness_checks.json", checks)
    dump(
        output / "provenance.json",
        {
            "git": git_provenance(),
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "k1_key": cache_meta["key"],
            "splits_manifest_sha256": sha256_file(ROOT / "research" / "manifests" / "splits.json"),
            "samples_manifest_sha256": sha256_file(ROOT / "research" / "manifests" / "samples.json"),
        },
    )
    dump(output / "status.json", {"status": "COMPLETED", "rule": "P01-R4", "target": name})
    result = {"target": name, "status": "COMPLETED", **summary, **resources, "timing": timing}
    dump(output / "result.json", result)
    return result


def main() -> None:
    config_path = ROOT / "configs" / "research" / "resource_controls.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    rows = []
    with gpu_lock():
        adapter, scene = load_reference(load_cameras=True)
        cache_meta = build_k1(adapter, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], ROOT / "research_cache" / "cut_roasted_beef")
        cache = torch.load(cache_meta["path"], map_location="cpu")
        build_verified_ground_truth_cache(adapter, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        checks = correctness_checks(adapter, cache)
        if not checks["passed"]:
            raise RuntimeError(checks)
        for target in config["targets"]:
            rows.append(run_target(target["name"], target_bytes(target), config["max_byte_probes"], adapter, scene, cache, cache_meta, splits, samples, checks))
    aggregate = {"status": "COMPLETED", "resource_control_slots_used": len(rows), "runs": rows}
    root = ROOT / "runs" / "research" / "resource_controls"
    dump(root / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
