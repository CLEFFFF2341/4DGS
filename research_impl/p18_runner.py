from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim

from .adapter import load_bundle, load_reference
from .config import ROOT, git_provenance, sha256_file
from .evaluate import (
    benchmark_latency,
    load_ground_truth,
    render_one,
    select_camera,
)
from .methods.p18 import (
    TemporalPlan,
    active_indices,
    build_plan,
    dp_toy,
    reconstruct_rle,
    rle_intervals,
    solve_chain,
)
from .p11_runner import load_cached_training_gt, prepare_training_frames
from .runner import gpu_lock


OUTPUT_ROOT = ROOT / "runs" / "research" / "P18" / "cut_roasted_beef" / "active0.5"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tensor_bytes(adapter: Any) -> int:
    seen = set()
    total = 0
    for value in vars(adapter.model).values():
        if isinstance(value, torch.Tensor) and value.data_ptr() not in seen:
            seen.add(value.data_ptr())
            total += value.numel() * value.element_size()
    return total


def find_k2(times: list[int], static_count: int, dynamic_count: int) -> Path:
    matches = []
    for path in (ROOT / "research_cache" / "cut_roasted_beef").glob("*/k2.pt"):
        value = torch.load(path, map_location="cpu")
        specification = value.get("specification", {})
        if (
            specification.get("schema") == "k2-exact-early-stop-deletion-v3"
            and specification.get("times") == times
            and value["static_rows"].numel() == static_count
            and value["dynamic_rows"].numel() == dynamic_count
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one full M0 K2-v3 cache, got {matches}")
    return matches[0]


@torch.no_grad()
def render_active(reference: Any, scene: Any, plan: TemporalPlan, name: str, timestamp: int):
    camera = select_camera(scene, name, timestamp)
    static, dynamic = active_indices(plan, timestamp, reference.static_count)
    gather_started = time.perf_counter()
    subset = reference.gather(static, dynamic)
    torch.cuda.synchronize()
    gather_seconds = time.perf_counter() - gather_started
    render_started = time.perf_counter()
    image = render_one(subset, camera)
    torch.cuda.synchronize()
    raster_seconds = time.perf_counter() - render_started
    counts = {"static": int(static.size), "dynamic": int(dynamic.size), "total": int(static.size + dynamic.size)}
    del subset
    return image, counts, gather_seconds, raster_seconds


@torch.no_grad()
def evaluate_active(
    reference: Any,
    scene: Any,
    plan: TemporalPlan,
    camera_names: list[str],
    times: list[int],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    lpips_model = LPIPS("alex", "0.1").cuda().eval()
    rows = []
    decoders = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for name in camera_names:
            for timestamp in times:
                camera = select_camera(scene, name, timestamp)
                gt = load_ground_truth(camera, decoders)
                image, counts, gather_seconds, raster_seconds = render_active(
                    reference, scene, plan, name, timestamp
                )
                teacher = render_one(reference, camera)
                method_mse = torch.mean((image - gt) ** 2)
                teacher_gt_mse = torch.mean((teacher - gt) ** 2)
                method_psnr = float(psnr(image[None], gt[None]).item())
                teacher_psnr = float(psnr(teacher[None], gt[None]).item())
                rows.append(
                    {
                        "camera": name,
                        "timestamp": timestamp,
                        "active_static": counts["static"],
                        "active_dynamic": counts["dynamic"],
                        "active_total": counts["total"],
                        "psnr": method_psnr,
                        "ssim": float(ssim(image[None], gt[None]).item()),
                        "lpips_alex": float(lpips_model(image[None], gt[None]).item()),
                        "gt_mse": float(method_mse.item()),
                        "teacher_mse": float(torch.mean((image - teacher) ** 2).item()),
                        "delta_mse": float((method_mse - teacher_gt_mse).item()),
                        "reference_psnr": teacher_psnr,
                        "psnr_drop": teacher_psnr - method_psnr,
                        "gather_seconds": gather_seconds,
                        "raster_seconds": raster_seconds,
                        "end_to_end_seconds": gather_seconds + raster_seconds,
                    }
                )
                del gt, image, teacher
    finally:
        for decoder in decoders.values():
            decoder.close()
    with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tail_count = max(1, int(np.ceil(0.1 * len(rows))))
    drop_descending = sorted(rows, key=lambda row: (-row["psnr_drop"], row["camera"], row["timestamp"]))
    last_rows = [row for row in rows if row["timestamp"] >= 270]
    summary = {
        "status": "COMPLETED",
        "count": len(rows),
        "mean_psnr": statistics.fmean(row["psnr"] for row in rows),
        "mean_ssim": statistics.fmean(row["ssim"] for row in rows),
        "mean_lpips_alex": statistics.fmean(row["lpips_alex"] for row in rows),
        "mean_gt_mse": statistics.fmean(row["gt_mse"] for row in rows),
        "mean_teacher_mse": statistics.fmean(row["teacher_mse"] for row in rows),
        "mean_delta_mse": statistics.fmean(row["delta_mse"] for row in rows),
        "mean_psnr_drop": statistics.fmean(row["psnr_drop"] for row in rows),
        "worst_10pct_mean_psnr_drop": statistics.fmean(
            row["psnr_drop"] for row in drop_descending[:tail_count]
        ),
        "p95_delta_mse": float(np.percentile([row["delta_mse"] for row in rows], 95)),
        "worst_frame": {
            "camera": drop_descending[0]["camera"],
            "timestamp": drop_descending[0]["timestamp"],
            "psnr_drop": drop_descending[0]["psnr_drop"],
        },
        "last_30_mean_psnr": statistics.fmean(row["psnr"] for row in last_rows),
        "last_30_mean_delta_mse": statistics.fmean(row["delta_mse"] for row in last_rows),
        "mean_active_static": statistics.fmean(row["active_static"] for row in rows),
        "mean_active_dynamic": statistics.fmean(row["active_dynamic"] for row in rows),
        "mean_active_total": statistics.fmean(row["active_total"] for row in rows),
        "mean_gather_seconds": statistics.fmean(row["gather_seconds"] for row in rows),
        "mean_raster_seconds": statistics.fmean(row["raster_seconds"] for row in rows),
        "mean_end_to_end_seconds": statistics.fmean(row["end_to_end_seconds"] for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    dump(output / "summary.json", summary)
    return summary


def _mean_masked(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return float(values[mask].mean())


@torch.no_grad()
def temporal_metrics(
    reference: Any,
    scene: Any,
    plan: TemporalPlan,
    windows: list[list[int]],
    camera_names: list[str],
    output: Path,
) -> dict[str, Any]:
    rows = []
    for name in camera_names:
        for window_start, window_end in windows:
            previous = None
            for timestamp in range(window_start, window_end + 1):
                camera = select_camera(scene, name, timestamp)
                gt = load_cached_training_gt(camera, name, timestamp)
                image, _, _, _ = render_active(reference, scene, plan, name, timestamp)
                teacher = render_one(reference, camera)
                current = {
                    "gt": gt,
                    "error": image - gt,
                    "reference_error": teacher - gt,
                    "teacher_residual": image - teacher,
                }
                if previous is not None:
                    error_delta = (current["error"] - previous["error"]).abs().mean(dim=0)
                    reference_delta = (
                        current["reference_error"] - previous["reference_error"]
                    ).abs().mean(dim=0)
                    teacher_delta = (
                        current["teacher_residual"] - previous["teacher_residual"]
                    ).abs().mean(dim=0)
                    gt_delta = (gt - previous["gt"]).abs().mean(dim=0)
                    motion = functional.max_pool2d(
                        (gt_delta > 0.03).float()[None, None], 3, stride=1, padding=1
                    )[0, 0].bool()
                    rows.append(
                        {
                            "camera": name,
                            "window_start": window_start,
                            "timestamp_previous": timestamp - 1,
                            "timestamp": timestamp,
                            "method_tde": float(error_delta.mean()),
                            "reference_tde": float(reference_delta.mean()),
                            "teacher_residual_tde": float(teacher_delta.mean()),
                            "gt_frame_difference": float(gt_delta.mean()),
                            "motion_fraction": float(motion.float().mean()),
                            "method_tde_motion_roi": _mean_masked(error_delta, motion),
                            "method_tde_nonmotion_roi": _mean_masked(error_delta, ~motion),
                            "reference_tde_motion_roi": _mean_masked(reference_delta, motion),
                            "reference_tde_nonmotion_roi": _mean_masked(reference_delta, ~motion),
                        }
                    )
                previous = current
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "pairs": len(rows),
        "method_tde": statistics.fmean(row["method_tde"] for row in rows),
        "reference_tde": statistics.fmean(row["reference_tde"] for row in rows),
        "incremental_tde": statistics.fmean(
            row["method_tde"] - row["reference_tde"] for row in rows
        ),
        "teacher_residual_tde": statistics.fmean(row["teacher_residual_tde"] for row in rows),
        "gt_frame_difference": statistics.fmean(row["gt_frame_difference"] for row in rows),
        "motion_fraction": statistics.fmean(row["motion_fraction"] for row in rows),
        "method_tde_motion_roi": statistics.fmean(
            row["method_tde_motion_roi"] for row in rows if row["method_tde_motion_roi"] is not None
        ),
        "method_tde_nonmotion_roi": statistics.fmean(
            row["method_tde_nonmotion_roi"] for row in rows if row["method_tde_nonmotion_roi"] is not None
        ),
    }
    dump(output.with_suffix(".summary.json"), result)
    return result


def switch_times(plan: TemporalPlan) -> list[int]:
    changed = np.any(plan.active[:, 1:] != plan.active[:, :-1], axis=0)
    starts = [int(np.flatnonzero(plan.bin_for_time == index)[0]) for index in range(1, plan.sample_times.size)]
    return [starts[index] for index in np.flatnonzero(changed)[:5]]


@torch.no_grad()
def switch_diagnostics(
    reference: Any,
    scene: Any,
    plan: TemporalPlan,
    camera_names: list[str],
    output: Path,
) -> dict[str, Any]:
    boundaries = switch_times(plan)
    times = sorted({max(0, min(299, boundary + delta)) for boundary in boundaries for delta in range(-2, 3)})
    rows = []
    for name in camera_names:
        for timestamp in times:
            camera = select_camera(scene, name, timestamp)
            gt = load_cached_training_gt(camera, name, timestamp)
            image, counts, _, _ = render_active(reference, scene, plan, name, timestamp)
            teacher = render_one(reference, camera)
            rows.append(
                {
                    "camera": name,
                    "timestamp": timestamp,
                    "active_total": counts["total"],
                    "psnr": float(psnr(image[None], gt[None]).item()),
                    "teacher_mse": float(torch.mean((image - teacher) ** 2)),
                }
            )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "boundaries": boundaries,
        "times": times,
        "frames": len(rows),
        "mean_psnr": statistics.fmean(row["psnr"] for row in rows),
        "mean_teacher_mse": statistics.fmean(row["teacher_mse"] for row in rows),
    }
    dump(output.with_suffix(".summary.json"), result)
    return result


@torch.no_grad()
def save_visualizations(reference: Any, scene: Any, plan: TemporalPlan, names: list[str], times: list[int], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        for timestamp in times:
            camera = select_camera(scene, name, timestamp)
            gt = load_cached_training_gt(camera, name, timestamp)
            teacher = render_one(reference, camera)
            image, _, _, _ = render_active(reference, scene, plan, name, timestamp)
            panel = torch.cat((gt, teacher, image), dim=2).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(panel).save(output / f"{name}_{timestamp:03d}_gt-reference-method.png")


@torch.no_grad()
def benchmark_active_latency(reference: Any, scene: Any, plan: TemporalPlan, inputs: list[tuple[str, int]], output: Path):
    cameras = [select_camera(scene, name, timestamp) for name, timestamp in inputs]

    def once(camera):
        static, dynamic = active_indices(plan, int(camera.timestamp), reference.static_count)
        total_start = torch.cuda.Event(enable_timing=True)
        gather_end = torch.cuda.Event(enable_timing=True)
        render_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        total_start.record()
        subset = reference.gather(static, dynamic)
        gather_end.record()
        render_one(subset, camera)
        render_end.record()
        torch.cuda.synchronize()
        values = (
            time.perf_counter() - wall_start,
            total_start.elapsed_time(render_end) / 1000.0,
            total_start.elapsed_time(gather_end) / 1000.0,
            gather_end.elapsed_time(render_end) / 1000.0,
        )
        del subset
        return values

    for index in range(20):
        once(cameras[index % len(cameras)])
    torch.cuda.reset_peak_memory_stats()
    blocks = []
    for block_index in range(3):
        rows = [once(cameras[index % len(cameras)]) for index in range(100)]
        wall, total_gpu, gather_gpu, raster_gpu = map(np.asarray, zip(*rows))
        blocks.append(
            {
                "block": block_index,
                "wall_mean_seconds": float(wall.mean()),
                "wall_p50_seconds": float(np.percentile(wall, 50)),
                "wall_p95_seconds": float(np.percentile(wall, 95)),
                "gpu_total_mean_seconds": float(total_gpu.mean()),
                "gpu_gather_mean_seconds": float(gather_gpu.mean()),
                "gpu_raster_mean_seconds": float(raster_gpu.mean()),
            }
        )
    result = {
        "status": "COMPLETED",
        "warmup": 20,
        "iterations_per_block": 100,
        "inputs": [{"camera": name, "timestamp": timestamp} for name, timestamp in inputs],
        "blocks": blocks,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    dump(output, result)
    return result


@torch.no_grad()
def shared_correctness(reference: Any, scene: Any, plans: dict[str, TemporalPlan], errors: np.ndarray) -> dict[str, Any]:
    all_indices_static = np.arange(reference.static_count, dtype=np.int64)
    all_indices_dynamic = np.arange(reference.dynamic_count, dtype=np.int64)
    full = reference.gather(all_indices_static, all_indices_dynamic)
    camera = select_camera(scene, "cam01", 149)
    identity_error = float((render_one(reference, camera) - render_one(full, camera)).abs().max())
    del full
    result = {
        "dp_toy": dp_toy(),
        "zero_endpoint_bin": int(plans["R0"].bin_for_time[0]) == 0,
        "last_endpoint_bin": int(plans["R0"].bin_for_time[299]) == 23,
        "bin_coverage_300": int(plans["R0"].bin_lengths.sum()) == 300,
        "allactive_render_max_abs": identity_error,
        "allactive_render_identity": identity_error <= 1e-6,
    }
    for rule, plan in plans.items():
        intervals = rle_intervals(plan)
        reconstructed = reconstruct_rle(plan.active.shape[0], plan.active.shape[1], intervals)
        result[f"{rule}_rle_consistent"] = bool(np.array_equal(reconstructed, plan.active))
        result[f"{rule}_under_cap"] = bool(
            plan.diagnostics["actual_active_frames"] <= plan.diagnostics["total_cap"]
        )
        static, dynamic = active_indices(plan, 149, reference.static_count)
        subset = reference.gather(static, dynamic)
        result[f"{rule}_concat_ids_correct"] = bool(
            np.array_equal(subset.static_rows, reference.static_rows[static])
            and np.array_equal(subset.dynamic_rows, reference.dynamic_rows[dynamic])
        )
        del subset
    r1 = plans["R1"]
    independent_static = errors[:, : reference.static_count].T > r1.lambdas["static"]
    independent_dynamic = errors[:, reference.static_count :].T > r1.lambdas["dynamic"]
    result["gamma0_independent_threshold"] = bool(
        np.array_equal(r1.active[: reference.static_count], independent_static)
        and np.array_equal(r1.active[reference.static_count :], independent_dynamic)
    )
    result["passed"] = bool(
        result["dp_toy"]["passed"]
        and all(value for key, value in result.items() if isinstance(value, bool))
    )
    return result


def save_activity(output: Path, plan: TemporalPlan) -> int:
    intervals = rle_intervals(plan)
    np.savez_compressed(
        output,
        **intervals,
        sample_times=plan.sample_times,
        bin_for_time=plan.bin_for_time,
        bin_lengths=plan.bin_lengths,
    )
    return output.stat().st_size


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs" / "research" / "p18.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        k2_path = find_k2(samples["T24"], reference.static_count, reference.dynamic_count)
        k2 = torch.load(k2_path, map_location="cpu")
        errors = k2["e_it"].numpy()
        plans = {}
        selection_seconds = {}
        for rule in config["rules"]:
            started = time.perf_counter()
            plans[rule] = build_plan(
                errors,
                np.asarray(samples["T24"], dtype=np.int64),
                reference.static_count,
                reference.dynamic_count,
                rule,
            )
            selection_seconds[rule] = time.perf_counter() - started
        checks = shared_correctness(reference, scene, plans, errors)
        repeated = {
            rule: [
                build_plan(
                    errors,
                    np.asarray(samples["T24"], dtype=np.int64),
                    reference.static_count,
                    reference.dynamic_count,
                    rule,
                )
                for _ in range(2)
            ]
            for rule in config["rules"]
        }
        checks["deterministic_seed_policy_0_1_2"] = all(
            np.array_equal(plans[rule].active, candidate.active)
            and plans[rule].lambdas == candidate.lambdas
            for rule in config["rules"]
            for candidate in repeated[rule]
        )
        checks["passed"] = bool(
            checks["passed"] and checks["deterministic_seed_policy_0_1_2"]
        )
        dump(OUTPUT_ROOT / "correctness_checks.json", checks)
        if not checks["passed"]:
            raise RuntimeError(checks)

        windows = config["temporal_windows"]
        frame_times = set()
        for start, end in windows:
            frame_times.update(range(start, end + 1))
        for plan in plans.values():
            for boundary in switch_times(plan):
                frame_times.update(max(0, min(299, boundary + delta)) for delta in range(-2, 3))
        frame_times.update(samples["fixed_visualization_times"])
        frame_schedule = [
            {"camera": name, "timestamp": timestamp}
            for name in splits["development"]
            for timestamp in sorted(frame_times)
        ]
        frame_cache = prepare_training_frames(scene, frame_schedule)
        dump(OUTPUT_ROOT / "temporal_frame_cache.json", frame_cache)

        latency_inputs = [
            (splits["development"][index % 2], samples["T24"][index]) for index in range(10)
        ]
        baseline_latency = benchmark_latency(reference, scene, latency_inputs, OUTPUT_ROOT / "baseline_latency.json")
        results = {}
        for rule, plan in plans.items():
            output = OUTPUT_ROOT / rule
            output.mkdir(parents=True, exist_ok=True)
            dump(output / "selection_trace.json", plan.traces)
            dump(output / "selection_diagnostics.json", {**plan.diagnostics, "lambdas": plan.lambdas})
            activity_bytes = save_activity(output / "activity_rle.npz", plan)
            bundle_dir = output / "bundle"
            bundle = reference.save_bundle(bundle_dir)
            restored = load_bundle(bundle_dir, reference.args)
            active_image, _, _, _ = render_active(reference, scene, plan, "cam01", 149)
            restored_image, _, _, _ = render_active(restored, scene, plan, "cam01", 149)
            reload_max_abs = float((active_image - restored_image).abs().max())
            if reload_max_abs > 1e-6:
                raise RuntimeError(f"{rule} bundle reload mismatch: {reload_max_abs}")
            smoke_rows = []
            torch.cuda.reset_peak_memory_stats()
            for timestamp in (0, 149):
                image, counts, gather_seconds, raster_seconds = render_active(
                    reference, scene, plan, splits["development"][0], timestamp
                )
                smoke_rows.append(
                    {
                        "camera": splits["development"][0],
                        "timestamp": timestamp,
                        "active": counts,
                        "finite": bool(torch.isfinite(image).all()),
                        "gather_seconds": gather_seconds,
                        "raster_seconds": raster_seconds,
                    }
                )
            smoke = {
                "status": "COMPLETED",
                "rows": smoke_rows,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            if not all(row["finite"] for row in smoke_rows):
                raise RuntimeError(smoke)
            dump(output / "smoke.json", smoke)
            run_started = time.perf_counter()
            summary = evaluate_active(
                reference,
                scene,
                plan,
                splits["development"],
                samples["T24"],
                output / "evaluation",
            )
            temporal = temporal_metrics(
                reference,
                scene,
                plan,
                windows,
                splits["development"],
                output / "temporal_metrics.csv",
            )
            switch = switch_diagnostics(
                reference,
                scene,
                plan,
                splits["development"],
                output / "switch_diagnostics.csv",
            )
            latency = benchmark_active_latency(
                reference, scene, plan, latency_inputs, output / "latency.json"
            )
            save_visualizations(
                reference,
                scene,
                plan,
                splits["development"],
                samples["fixed_visualization_times"],
                output / "visualizations",
            )
            resources = {
                "unique_points": reference.total_count,
                "static_points": reference.static_count,
                "dynamic_points": reference.dynamic_count,
                "actual_active_frames": plan.diagnostics["actual_active_frames"],
                "activity_cap": plan.diagnostics["total_cap"],
                "loaded_tensor_bytes": tensor_bytes(reference),
                "bundle_bytes": directory_bytes(bundle_dir),
                "activity_rle_bytes": activity_bytes,
                "deployable_package_bytes": directory_bytes(bundle_dir) + activity_bytes,
                "smoke": smoke,
                "latency": latency,
            }
            timing = {
                "selection_seconds": selection_seconds[rule],
                "run_seconds": time.perf_counter() - run_started,
            }
            dump(output / "resolved_config.json", {**config, "rule": rule})
            dump(output / "bundle_reload_check.json", {"render_max_abs": reload_max_abs, "passed": True})
            dump(output / "resources.json", resources)
            dump(output / "timing.json", timing)
            dump(
                output / "provenance.json",
                {
                    "git": git_provenance(),
                    "checkpoint_sha256": reference.checkpoint_sha256,
                    "k2_sha256": sha256_file(k2_path),
                    "config_sha256": sha256_file(config_path),
                },
            )
            (output / "commands.txt").write_text(
                f"python -m research_impl.p18_runner  # {rule}\n", encoding="utf-8"
            )
            result = {
                "status": "COMPLETED",
                "rule": rule,
                "summary": summary,
                "temporal": temporal,
                "switch": switch,
                "resources": resources,
                "bundle": bundle,
            }
            dump(output / "result.json", result)
            dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
            results[rule] = result
            del restored
        r0_tde = results["R0"]["temporal"]["method_tde"]
        r1_tde = results["R1"]["temporal"]["method_tde"]
        baseline_wall = statistics.fmean(
            block["wall_mean_seconds"] for block in baseline_latency["blocks"]
        )
        aggregate = {
            "status": "COMPLETED",
            "seed_policy": "deterministic activity verified for seeds 0,1,2; one quality evaluation per rule",
            "correctness": checks,
            "selection_seconds": selection_seconds,
            "baseline_latency": baseline_latency,
            "R0": results["R0"],
            "R1": results["R1"],
            "r0_tde_reduction_fraction_vs_r1": (r1_tde - r0_tde) / r1_tde,
            "r0_psnr_minus_r1": results["R0"]["summary"]["mean_psnr"]
            - results["R1"]["summary"]["mean_psnr"],
            "baseline_wall_mean_seconds": baseline_wall,
            "r0_wall_speedup": baseline_wall
            / statistics.fmean(
                block["wall_mean_seconds"] for block in results["R0"]["resources"]["latency"]["blocks"]
            ),
            "r1_wall_speedup": baseline_wall
            / statistics.fmean(
                block["wall_mean_seconds"] for block in results["R1"]["resources"]["latency"]["blocks"]
            ),
        }
        dump(OUTPUT_ROOT / "aggregate.json", aggregate)
        print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
