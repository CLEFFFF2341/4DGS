from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapter import load_bundle, load_reference
from .cache import build_k1
from .config import ROOT, git_provenance, sha256_file, sha256_json
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p01 import RULES, correctness_checks, select
from .runner import gpu_lock


RUN_ROOT = ROOT / "runs" / "research" / "P01" / "cut_roasted_beef" / "b0.5" / "seed0"
CONFIG_PATH = ROOT / "configs" / "research" / "p01.json"
MANIFEST_ROOT = ROOT / "research" / "manifests"
PROGRESS_PATH = ROOT / "research" / "progress.json"
CACHE_ROOT = ROOT / "research_cache" / "cut_roasted_beef"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _tensor_bytes(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for value in vars(model).values():
        if isinstance(value, torch.Tensor) and value.data_ptr() not in seen:
            seen.add(value.data_ptr())
            total += value.numel() * value.element_size()
    return total


def _method_report(rule: str, summary: dict[str, Any], selection: Any, cache_key: str) -> str:
    return f"""# P01 {rule} 方法报告

状态：`COMPLETED`。这是零微调、50% 数量预算的预注册简单基线；不主张新颖性。

- 实际点数：{selection.diagnostics['actual_total']}（static {selection.static_indices.size} / dynamic {selection.dynamic_indices.size}）
- K1 cache：`{cache_key}`
- mean PSNR：{summary['mean_psnr']:.6f} dB
- mean PSNR drop：{summary['mean_psnr_drop']:.6f} dB
- worst-10% mean drop：{summary['worst_10pct_mean_psnr_drop']:.6f} dB
- LPIPS-Alex：{summary['mean_lpips_alex']:.6f}
- teacher MSE：{summary['mean_teacher_mse']:.9g}

近似与风险：R1/R2/R3/R4 都只利用冻结参考模型的训练视角统计；贡献高并不等价于联合删除误差低。R4 的体积因子可能偏好大点。R5 是 R2 权重的指数竞赛无放回采样。所有规则固定 static/dynamic 配额，未使用开发集调分数或参数。
"""


def _bundle_reload_check(directory: Path, adapter: Any, scene: Any, camera_name: str) -> dict[str, Any]:
    kept = np.load(directory / "kept_ids.npz")
    expected = adapter.gather(kept["static_indices"], kept["dynamic_indices"])
    reloaded = load_bundle(directory / "bundle", adapter.args)
    camera = select_camera(scene, camera_name, 149)
    expected_image = render_one(expected, camera)
    reloaded_image = render_one(reloaded, camera)
    difference = (expected_image - reloaded_image).abs()
    result = {
        "static_rows_equal": bool(np.array_equal(expected.static_rows, reloaded.static_rows)),
        "dynamic_rows_equal": bool(np.array_equal(expected.dynamic_rows, reloaded.dynamic_rows)),
        "render_max_abs": float(difference.max()),
        "render_mean_abs": float(difference.mean()),
    }
    result["passed"] = result["static_rows_equal"] and result["dynamic_rows_equal"] and result["render_max_abs"] <= 1e-6
    _write_json(directory / "bundle_reload_check.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Bundle reload check failed for {directory}: {result}")
    return result


def _run_one(
    rule: str,
    config: dict[str, Any],
    adapter: Any,
    scene: Any,
    cache: dict[str, Any],
    cache_meta: dict[str, Any],
    splits: dict[str, Any],
    samples: dict[str, Any],
    checks: dict[str, Any],
    selection_result: Any | None = None,
    method_report_text: str | None = None,
) -> dict[str, Any]:
    resolved = {**config, "rule": rule, "cache_key": cache_meta["key"]}
    config_hash = sha256_json(resolved)
    final = RUN_ROOT / config_hash
    status_path = final / "status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "COMPLETED":
        _bundle_reload_check(final, adapter, scene, splits["development"][0])
        summary = json.loads((final / "summary.json").read_text(encoding="utf-8"))
        return {
            **summary,
            "rule": rule,
            "config_hash": config_hash,
            "resources": json.loads((final / "resources.json").read_text(encoding="utf-8")),
            "timing": json.loads((final / "timing.json").read_text(encoding="utf-8")),
        }
    temporary = RUN_ROOT / f".{config_hash}.partial-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir(parents=True, exist_ok=False)
    _write_json(temporary / "resolved_config.json", resolved)
    _write_json(temporary / "status.json", {"status": "RUNNING", "rule": rule})
    _write_json(temporary / "correctness_checks.json", checks)
    (temporary / "commands.txt").write_text(
        "python -m research_impl.p01_runner\n",
        encoding="utf-8",
    )
    timing: dict[str, float] = {}
    started = time.perf_counter()
    selection_started = time.perf_counter()
    selection = selection_result if selection_result is not None else select(adapter, cache, rule, config["budget_fraction"], config["seed"])
    timing["selection_seconds"] = time.perf_counter() - selection_started
    np.savez_compressed(
        temporary / "kept_ids.npz",
        static_rows=adapter.static_rows[selection.static_indices],
        dynamic_rows=adapter.dynamic_rows[selection.dynamic_indices],
        static_indices=selection.static_indices,
        dynamic_indices=selection.dynamic_indices,
    )
    score_payload = {
        "rule": rule,
        "static_scores": selection.static_scores,
        "dynamic_scores": selection.dynamic_scores,
        "diagnostics": selection.diagnostics,
    }
    torch.save(score_payload, temporary / "selection.pt")
    _write_json(temporary / "selection.json", selection.diagnostics)
    subset = adapter.gather(selection.static_indices, selection.dynamic_indices)
    bundle_started = time.perf_counter()
    bundle_meta = subset.save_bundle(temporary / "bundle")
    _bundle_reload_check(temporary, adapter, scene, splits["development"][0])
    timing["serialization_seconds"] = time.perf_counter() - bundle_started
    smoke_started = time.perf_counter()
    smoke = smoke_render(subset, scene, [splits["development"][0]], [0, 149], temporary / "smoke")
    timing["smoke_seconds"] = time.perf_counter() - smoke_started
    evaluation_started = time.perf_counter()
    summary = evaluate_against_reference(
        subset,
        adapter,
        scene,
        splits["development"],
        samples["T24"],
        temporary,
    )
    timing["evaluation_seconds"] = time.perf_counter() - evaluation_started
    latency_inputs = [
        (splits["development"][index % len(splits["development"])], samples["T24"][index])
        for index in range(10)
    ]
    latency_started = time.perf_counter()
    latency = benchmark_latency(subset, scene, latency_inputs, temporary / "latency.json")
    timing["latency_seconds"] = time.perf_counter() - latency_started
    visualization_started = time.perf_counter()
    save_comparison_visualizations(
        subset,
        adapter,
        scene,
        splits["development"],
        samples["fixed_visualization_times"],
        temporary / "visualizations",
    )
    timing["visualization_seconds"] = time.perf_counter() - visualization_started
    timing["total_seconds"] = time.perf_counter() - started
    provenance = {
        "git": git_provenance(),
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "cache_key": cache_meta["key"],
        "cache_path": cache_meta["path"],
        "reference_manifest_sha256": sha256_file(MANIFEST_ROOT / "reference.json"),
        "splits_manifest_sha256": sha256_file(MANIFEST_ROOT / "splits.json"),
        "samples_manifest_sha256": sha256_file(MANIFEST_ROOT / "samples.json"),
    }
    resources = {
        "actual_total_points": selection.diagnostics["actual_total"],
        "static_points": int(selection.static_indices.size),
        "dynamic_points": int(selection.dynamic_indices.size),
        "loaded_tensor_bytes": _tensor_bytes(subset.model),
        "bundle_bytes": _directory_bytes(temporary / "bundle"),
        "peak_allocated_bytes": max(smoke["peak_allocated_bytes"], summary["peak_allocated_bytes"]),
        "peak_reserved_bytes": max(smoke["peak_reserved_bytes"], summary["peak_reserved_bytes"]),
        "latency": latency,
    }
    _write_json(temporary / "provenance.json", provenance)
    _write_json(temporary / "timing.json", timing)
    _write_json(temporary / "resources.json", resources)
    _write_json(temporary / "bundle_metadata.json", bundle_meta)
    (temporary / "method_report.md").write_text(
        method_report_text or _method_report(rule, summary, selection, cache_meta["key"]), encoding="utf-8"
    )
    _write_json(temporary / "status.json", {"status": "COMPLETED", "rule": rule, "config_hash": config_hash})
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return {**summary, "rule": rule, "config_hash": config_hash, "resources": resources, "timing": timing}


def _overlaps(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids: dict[str, set[tuple[str, int]]] = {}
    for row in run_rows:
        stable = np.load(RUN_ROOT / row["config_hash"] / "kept_ids.npz")
        ids[row["rule"]] = {
            *(("static", int(value)) for value in stable["static_rows"]),
            *(("dynamic", int(value)) for value in stable["dynamic_rows"]),
        }
    matrix = {}
    for left in RULES:
        matrix[left] = {}
        for right in RULES:
            intersection = len(ids[left] & ids[right])
            union = len(ids[left] | ids[right])
            matrix[left][right] = intersection / union if union else 1.0
    return matrix


def _update_progress(rows: list[dict[str, Any]]) -> None:
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    progress["runs"]["P01/R0-R5"] = "COMPLETED"
    progress["queue"] = [{"task": "P01", "status": "COMPLETED"}]
    progress["gpu_hours_consumed"] = round(
        float(progress.get("gpu_hours_consumed", 0.0))
        + sum(row.get("timing", {}).get("total_seconds", 0.0) for row in rows) / 3600.0,
        4,
    )
    progress["git"] = git_provenance()
    _write_json(PROGRESS_PATH, progress)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    splits = json.loads((MANIFEST_ROOT / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((MANIFEST_ROOT / "samples.json").read_text(encoding="utf-8"))
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        adapter, scene = load_reference(load_cameras=True)
        cache_meta = build_k1(adapter, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], CACHE_ROOT)
        cache = torch.load(cache_meta["path"], map_location="cpu")
        verified_gt = build_verified_ground_truth_cache(
            adapter,
            scene,
            splits["development"],
            samples["T24"],
            ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv",
        )
        checks = correctness_checks(adapter, cache)
        if not checks["passed"]:
            raise RuntimeError(f"P01 correctness checks failed: {checks}")
        rows = []
        for rule in config["rules"]:
            print(f"P01 {rule}: start", flush=True)
            row = _run_one(rule, config, adapter, scene, cache, cache_meta, splits, samples, checks)
            rows.append(row)
            print(f"P01 {rule}: PSNR={row['mean_psnr']:.6f}, drop={row['mean_psnr_drop']:.6f}", flush=True)
    overlap = _overlaps(rows)
    winner = max(rows, key=lambda row: (row["mean_psnr"], -row["mean_lpips_alex"], row["rule"]))
    aggregate = {
        "status": "COMPLETED",
        "cache_key": cache_meta["key"],
        "verified_gt": verified_gt,
        "correctness_checks": checks,
        "runs": rows,
        "jaccard_overlap": overlap,
        "best_simple_rule_on_development": winner["rule"],
    }
    _write_json(RUN_ROOT / "aggregate.json", aggregate)
    with (RUN_ROOT / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["rule", "mean_psnr", "mean_ssim", "mean_lpips_alex", "mean_teacher_mse", "mean_psnr_drop", "worst_10pct_mean_psnr_drop", "config_hash"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    _update_progress(rows)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
