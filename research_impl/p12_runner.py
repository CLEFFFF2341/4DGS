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
from .methods.p12 import correctness_checks, select
from .runner import gpu_lock


OUTPUT_ROOT = ROOT / "runs" / "research" / "P12" / "cut_roasted_beef" / "b0.5"
P02_ROOT = ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def reload_check(expected, restored, scene, camera_name: str) -> dict[str, Any]:
    camera = select_camera(scene, camera_name, 149)
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


def run_rule(reference, scene, scores, splits, samples, config, rule: str) -> dict[str, Any]:
    output = OUTPUT_ROOT / rule
    status_path = output / "status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "COMPLETED":
        return json.loads((output / "result.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    dump(status_path, {"status": "RUNNING", "rule": rule})
    started = time.perf_counter()
    selection_started = time.perf_counter()
    selection = select(reference, scores, rule, float(config["budget_fraction"]))
    selection_seconds = time.perf_counter() - selection_started
    subset = reference.gather(selection.static_indices, selection.dynamic_indices)
    np.savez_compressed(
        output / "kept_ids.npz",
        static_indices=selection.static_indices,
        dynamic_indices=selection.dynamic_indices,
        static_rows=reference.static_rows[selection.static_indices],
        dynamic_rows=reference.dynamic_rows[selection.dynamic_indices],
    )
    dump(output / "selection.json", selection.diagnostics)
    bundle_dir = output / "bundle"
    if not bundle_dir.exists():
        bundle = subset.save_bundle(bundle_dir)
    else:
        bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    roundtrip = reload_check(subset, load_bundle(bundle_dir, reference.args), scene, splits["development"][0])
    if not roundtrip["passed"]:
        raise RuntimeError(roundtrip)
    smoke = smoke_render(subset, scene, [splits["development"][0]], [0, 149], output / "smoke")
    summary = evaluate_against_reference(
        subset, reference, scene, splits["development"], samples["T24"], output / "evaluation"
    )
    inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
    latency = benchmark_latency(subset, scene, inputs, output / "latency.json")
    save_comparison_visualizations(
        subset,
        reference,
        scene,
        splits["development"],
        samples["fixed_visualization_times"],
        output / "visualizations",
    )
    total_seconds = time.perf_counter() - started
    resources = {
        "actual_total_points": subset.total_count,
        "static_points": subset.static_count,
        "dynamic_points": subset.dynamic_count,
        "bundle_bytes": directory_bytes(bundle_dir),
        "smoke": smoke,
        "latency": latency,
    }
    dump(output / "resolved_config.json", {**config, "rule": rule})
    dump(output / "bundle_reload_check.json", roundtrip)
    dump(output / "timing.json", {"selection_seconds": selection_seconds, "total_seconds": total_seconds})
    dump(output / "resources.json", resources)
    dump(
        output / "provenance.json",
        {
            "git": git_provenance(),
            "checkpoint_sha256": reference.checkpoint_sha256,
            "p02_selection_sha256": sha256_file(P02_ROOT / "selection.pt"),
            "config_sha256": sha256_file(ROOT / "configs" / "research" / "p12.json"),
        },
    )
    (output / "commands.txt").write_text(f"python -m research_impl.p12_runner  # {rule}\n", encoding="utf-8")
    result = {
        "status": "COMPLETED",
        "rule": rule,
        "selection": selection.diagnostics,
        "summary": summary,
        "resources": resources,
        "bundle": bundle,
    }
    dump(output / "result.json", result)
    dump(status_path, {"status": "COMPLETED", "rule": rule})
    return result


def main() -> None:
    config_path = ROOT / "configs" / "research" / "p12.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    p02_state = torch.load(P02_ROOT / "selection.pt", map_location="cpu")
    scores = np.asarray(p02_state["scores"], dtype=np.float64)
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        checks = correctness_checks(reference, scores)
        dump(OUTPUT_ROOT / "correctness_checks.json", checks)
        if not checks["passed"]:
            raise RuntimeError(checks)
        results = [run_rule(reference, scene, scores, splits, samples, config, rule) for rule in config["rules"]]
    p02_summary = json.loads((P02_ROOT / "summary.json").read_text(encoding="utf-8"))
    aggregate = {
        "status": "COMPLETED",
        "seed_policy": "deterministic IDs verified for seeds 0,1,2; one quality evaluation per rule",
        "p02": p02_summary,
        "rules": {result["rule"]: result for result in results},
    }
    for result in results:
        result["delta_psnr_vs_r0"] = result["summary"]["mean_psnr"] - results[0]["summary"]["mean_psnr"]
        result["delta_bytes_vs_r0"] = result["resources"]["bundle_bytes"] - results[0]["resources"]["bundle_bytes"]
    dump(OUTPUT_ROOT / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
