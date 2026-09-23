from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .adapter import load_bundle, load_reference
from .cache import build_k2
from .config import ROOT, git_provenance, sha256_file
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p02 import correctness_checks, select
from .runner import gpu_lock


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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


def main() -> None:
    config_path = ROOT / "configs" / "research" / "p02.json"
    splits_path = ROOT / "research" / "manifests" / "splits.json"
    samples_path = ROOT / "research" / "manifests" / "samples.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    output = ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0"
    output.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        k2_meta = build_k2(reference, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], ROOT / "research_cache" / "cut_roasted_beef")
        k2 = torch.load(k2_meta["path"], map_location="cpu")
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        checks = correctness_checks(reference, k2)
        if not checks["passed"]:
            raise RuntimeError(checks)
        started = time.perf_counter()
        selection_started = time.perf_counter()
        selection = select(reference, k2, config["budget_fraction"])
        selection_seconds = time.perf_counter() - selection_started
        subset = reference.gather(selection.static_indices, selection.dynamic_indices)
        np.savez_compressed(
            output / "kept_ids.npz",
            static_indices=selection.static_indices,
            dynamic_indices=selection.dynamic_indices,
            static_rows=reference.static_rows[selection.static_indices],
            dynamic_rows=reference.dynamic_rows[selection.dynamic_indices],
        )
        torch.save({"scores": torch.from_numpy(selection.scores), "diagnostics": selection.diagnostics}, output / "selection.pt")
        dump(output / "selection.json", selection.diagnostics)
        bundle_dir = output / "bundle"
        if not bundle_dir.exists():
            subset.save_bundle(bundle_dir)
        roundtrip = reload_check(subset, load_bundle(bundle_dir, reference.args), scene, splits["development"][0])
        if not roundtrip["passed"]:
            raise RuntimeError(roundtrip)
        render_smoke = smoke_render(subset, scene, [splits["development"][0]], [0, 149], output / "smoke")
        summary = evaluate_against_reference(subset, reference, scene, splits["development"], samples["T24"], output)
        inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
        latency = benchmark_latency(subset, scene, inputs, output / "latency.json")
        save_comparison_visualizations(subset, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
        total_seconds = time.perf_counter() - started
    dump(output / "resolved_config.json", {**config, "k2_key": k2_meta["key"]})
    (output / "commands.txt").write_text("python -m research_impl.p02_runner\n", encoding="utf-8")
    dump(output / "correctness_checks.json", checks)
    dump(output / "bundle_reload_check.json", roundtrip)
    dump(output / "timing.json", {"selection_seconds": selection_seconds, "total_seconds": total_seconds})
    dump(
        output / "resources.json",
        {
            "actual_total_points": selection.diagnostics["actual_total"],
            "static_points": int(selection.static_indices.size),
            "dynamic_points": int(selection.dynamic_indices.size),
            "bundle_bytes": directory_bytes(bundle_dir),
            "render_smoke": render_smoke,
            "latency": latency,
        },
    )
    dump(
        output / "provenance.json",
        {
            "git": git_provenance(),
            "checkpoint_sha256": reference.checkpoint_sha256,
            "k2_key": k2_meta["key"],
            "config_sha256": sha256_file(config_path),
            "splits_manifest_sha256": sha256_file(splits_path),
            "samples_manifest_sha256": sha256_file(samples_path),
        },
    )
    dump(output / "status.json", {"status": "COMPLETED", "rule": "P02-R0"})
    result = {"status": "COMPLETED", "k2_key": k2_meta["key"], "correctness_checks": checks, "summary": summary, "selection": selection.diagnostics}
    dump(output.parent.parent / "aggregate.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
