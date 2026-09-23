from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .adapter import load_bundle, load_reference
from .cache import build_k2
from .config import ROOT, git_provenance, sha256_file
from .evaluate import VideoFrameDecoder, benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, load_ground_truth, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p03 import correctness_checks, select
from .runner import gpu_lock


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def temporal_diagnostics(method, reference, scene, camera_names: list[str], windows: list[list[int]]) -> dict:
    metrics = {"method_tde": [], "reference_tde": [], "teacher_residual_tde": [], "gt_frame_difference": [], "motion_roi_tde": [], "static_roi_tde": []}
    pairs = []
    for name in camera_names:
        decoders = {}
        try:
            for window_start, window_end in windows:
                previous = None
                for timestamp in range(window_start, window_end + 1):
                    camera = select_camera(scene, name, timestamp)
                    ground_truth = load_ground_truth(camera, decoders)
                    method_image = render_one(method, camera)
                    reference_image = render_one(reference, camera)
                    state = {
                        "timestamp": timestamp,
                        "gt": ground_truth,
                        "method_residual": method_image - ground_truth,
                        "reference_residual": reference_image - ground_truth,
                        "teacher_residual": method_image - reference_image,
                    }
                    if previous is not None:
                        method_change = (state["method_residual"] - previous["method_residual"]).abs().mean(dim=0)
                        reference_change = (state["reference_residual"] - previous["reference_residual"]).abs().mean(dim=0)
                        teacher_change = (state["teacher_residual"] - previous["teacher_residual"]).abs().mean(dim=0)
                        gt_change = (state["gt"] - previous["gt"]).abs().mean(dim=0)
                        roi = functional.max_pool2d((gt_change > 0.03).float()[None, None], 3, stride=1, padding=1)[0, 0].bool()
                        metrics["method_tde"].append(float(method_change.mean()))
                        metrics["reference_tde"].append(float(reference_change.mean()))
                        metrics["teacher_residual_tde"].append(float(teacher_change.mean()))
                        metrics["gt_frame_difference"].append(float(gt_change.mean()))
                        metrics["motion_roi_tde"].append(float(method_change[roi].mean()) if bool(roi.any()) else 0.0)
                        metrics["static_roi_tde"].append(float(method_change[~roi].mean()) if bool((~roi).any()) else 0.0)
                        pairs.append({"camera": name, "from": previous["timestamp"], "to": timestamp, "motion_fraction": float(roi.float().mean())})
                    previous = state
        finally:
            for decoder in decoders.values():
                decoder.close()
    result = {key: float(np.mean(values)) for key, values in metrics.items()}
    result["incremental_tde"] = result["method_tde"] - result["reference_tde"]
    result["pair_count"] = len(pairs)
    result["pairs"] = pairs
    result["roi_definition"] = "mean RGB GT frame difference >0.03, dilated 3x3; motion approximation, not semantic truth"
    return result


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
    config_path = ROOT / "configs" / "research" / "p03.json"
    splits_path = ROOT / "research" / "manifests" / "splits.json"
    samples_path = ROOT / "research" / "manifests" / "samples.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P03" / "cut_roasted_beef" / "b0.5"
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        k2_meta = build_k2(reference, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], ROOT / "research_cache" / "cut_roasted_beef")
        k2 = torch.load(k2_meta["path"], map_location="cpu")
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        checks = correctness_checks(reference, k2)
        if not checks["passed"]:
            raise RuntimeError(checks)
        for rule in config["rules"]:
            output = base / rule
            output.mkdir(parents=True, exist_ok=True)
            selection = select(reference, k2, rule, config["budget_fraction"])
            if rule == "R0":
                p02 = ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0"
                subset = load_bundle(p02 / "bundle", reference.args)
                summary = json.loads((p02 / "summary.json").read_text(encoding="utf-8"))
                reuse = {"reused_from": str(p02), "formal_quality_run_reused": True}
                if not (np.array_equal(subset.static_rows, reference.static_rows[selection.static_indices]) and np.array_equal(subset.dynamic_rows, reference.dynamic_rows[selection.dynamic_indices])):
                    raise RuntimeError("P03 R0 selection does not match P02")
            else:
                subset = reference.gather(selection.static_indices, selection.dynamic_indices)
                bundle_dir = output / "bundle"
                if not bundle_dir.exists():
                    subset.save_bundle(bundle_dir)
                roundtrip = reload_check(subset, load_bundle(bundle_dir, reference.args), scene, splits["development"][0])
                if not roundtrip["passed"]:
                    raise RuntimeError(roundtrip)
                smoke_render(subset, scene, [splits["development"][0]], [0, 149], output / "smoke")
                summary = evaluate_against_reference(subset, reference, scene, splits["development"], samples["T24"], output)
                inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
                benchmark_latency(subset, scene, inputs, output / "latency.json")
                save_comparison_visualizations(subset, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
                reuse = {"formal_quality_run_reused": False, "roundtrip": roundtrip, "bundle_bytes": directory_bytes(bundle_dir)}
            temporal_started = time.perf_counter()
            temporal = temporal_diagnostics(subset, reference, scene, splits["development"], samples["windows"])
            temporal["seconds"] = time.perf_counter() - temporal_started
            dump(output / "temporal_diagnostics.json", temporal)
            np.savez_compressed(output / "kept_ids.npz", static_indices=selection.static_indices, dynamic_indices=selection.dynamic_indices, static_rows=reference.static_rows[selection.static_indices], dynamic_rows=reference.dynamic_rows[selection.dynamic_indices])
            dump(output / "selection.json", selection.diagnostics)
            dump(output / "resolved_config.json", {**config, "rule": rule, "k2_key": k2_meta["key"]})
            dump(output / "correctness_checks.json", checks)
            dump(output / "resources.json", reuse)
            dump(output / "provenance.json", {"git": git_provenance(), "checkpoint_sha256": reference.checkpoint_sha256, "k2_key": k2_meta["key"], "config_sha256": sha256_file(config_path), "splits_manifest_sha256": sha256_file(splits_path), "samples_manifest_sha256": sha256_file(samples_path)})
            (output / "commands.txt").write_text("python -m research_impl.p03_runner\n", encoding="utf-8")
            dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
            rows.append({"rule": rule, **summary, "selection": selection.diagnostics, "temporal": {key: value for key, value in temporal.items() if key != "pairs"}, **reuse})
            print(f"P03 {rule}: PSNR={summary['mean_psnr']:.6f}, incremental_TDE={temporal['incremental_tde']:.8f}", flush=True)
    aggregate = {"status": "COMPLETED", "k2_key": k2_meta["key"], "correctness_checks": checks, "runs": rows}
    dump(base.parent / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
