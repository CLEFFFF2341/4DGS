from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .adapter import load_bundle, load_reference
from .cache import build_k1, build_k2
from .config import ROOT, git_provenance, sha256_file
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p01 import compute_scores
from .methods.p06 import correctness_checks, select
from .p03_runner import build_consensus_window_cache, temporal_diagnostics
from .runner import gpu_lock


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def reload_check(expected, restored, scene, camera_name: str) -> dict:
    camera = select_camera(scene, camera_name, 149)
    difference = (render_one(expected, camera) - render_one(restored, camera)).abs()
    result = {"static_rows_equal": bool(np.array_equal(expected.static_rows, restored.static_rows)), "dynamic_rows_equal": bool(np.array_equal(expected.dynamic_rows, restored.dynamic_rows)), "render_max_abs": float(difference.max()), "render_mean_abs": float(difference.mean())}
    result["passed"] = result["static_rows_equal"] and result["dynamic_rows_equal"] and result["render_max_abs"] <= 1e-6
    return result


def segment_metrics(per_frame_csv: Path, times: list[int]) -> list[dict]:
    with per_frame_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {timestamp: position for position, timestamp in enumerate(times)}
    result = []
    for segment in range(6):
        allowed = {timestamp for timestamp, position in index.items() if position // 4 == segment}
        chosen = [row for row in rows if int(row["timestamp"]) in allowed]
        result.append({
            "segment": segment,
            "timestamps": sorted(allowed),
            "mean_delta_mse": float(np.mean([float(row["delta_mse"]) for row in chosen])),
            "mean_psnr_drop": float(np.mean([float(row["psnr_drop"]) for row in chosen])),
        })
    return result


def main() -> None:
    config_path = ROOT / "configs" / "research" / "p06.json"
    splits_path = ROOT / "research" / "manifests" / "splits.json"
    samples_path = ROOT / "research" / "manifests" / "samples.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P06" / "cut_roasted_beef" / "b0.5"
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        k2_meta = build_k2(reference, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], ROOT / "research_cache" / "cut_roasted_beef")
        k2 = torch.load(k2_meta["path"], map_location="cpu")
        k1_meta = build_k1(reference, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], ROOT / "research_cache" / "cut_roasted_beef")
        k1 = torch.load(k1_meta["path"], map_location="cpu")
        fallback_scores = compute_scores(k1)["R2"]
        checks = correctness_checks(reference, k2, fallback_scores)
        if not checks["passed"]:
            raise RuntimeError(checks)
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        build_consensus_window_cache(scene, splits["development"], samples["windows"])
        for rule in config["rules"]:
            output = base / rule
            output.mkdir(parents=True, exist_ok=True)
            selection = select(reference, k2, rule, config["budget_fraction"], config["multiplicative_weight_rounds"], config["multiplicative_weight_eta"], fallback_scores)
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
            latency = benchmark_latency(subset, scene, inputs, output / "latency.json")
            save_comparison_visualizations(subset, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
            temporal = temporal_diagnostics(subset, reference, scene, splits["development"], samples["windows"])
            segments = segment_metrics(output / "per_frame.csv", samples["T24"])
            winner_z = np.asarray(selection.diagnostics["winner_z"])
            delta = np.asarray([item["mean_delta_mse"] for item in segments])
            proxy_correlation = float(spearmanr(winner_z, -delta).statistic)
            diagnostics = {**selection.diagnostics, "true_segment_metrics": segments, "z_vs_negative_delta_mse_spearman": proxy_correlation}
            np.savez_compressed(output / "kept_ids.npz", static_indices=selection.static_indices, dynamic_indices=selection.dynamic_indices, static_rows=reference.static_rows[selection.static_indices], dynamic_rows=reference.dynamic_rows[selection.dynamic_indices])
            dump(output / "selection.json", diagnostics)
            dump(output / "temporal_diagnostics.json", temporal)
            dump(output / "resolved_config.json", {**config, "rule": rule, "k2_key": k2_meta["key"]})
            dump(output / "correctness_checks.json", checks)
            dump(output / "resources.json", {"bundle_bytes": directory_bytes(bundle_dir), "roundtrip": roundtrip, "latency": latency})
            dump(output / "provenance.json", {"git": git_provenance(), "checkpoint_sha256": reference.checkpoint_sha256, "k2_key": k2_meta["key"], "config_sha256": sha256_file(config_path), "splits_manifest_sha256": sha256_file(splits_path), "samples_manifest_sha256": sha256_file(samples_path)})
            (output / "commands.txt").write_text("python -m research_impl.p06_runner\n", encoding="utf-8")
            dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
            rows.append({"rule": rule, **summary, "selection": diagnostics, "temporal": {key: value for key, value in temporal.items() if key != "pairs"}})
            print(f"P06 {rule}: PSNR={summary['mean_psnr']:.6f}, tail={summary['worst_10pct_mean_psnr_drop']:.6f}, proxy_rho={proxy_correlation:.4f}", flush=True)
    aggregate = {"status": "COMPLETED", "k2_key": k2_meta["key"], "correctness_checks": checks, "runs": rows}
    dump(base.parent / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
