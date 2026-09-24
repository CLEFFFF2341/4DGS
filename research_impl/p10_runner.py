from __future__ import annotations

import copy
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .adapter import load_bundle, load_reference
from .cache import build_k1, build_k2, build_k4
from .config import ROOT, git_provenance, sha256_file
from .evaluate import VERIFIED_GT_ROOT, VideoFrameDecoder, benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, load_ground_truth, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p10 import attach_feasible_candidates, correctness_checks, enumerate_reappearing_demands, project_centers, repair_selection
from .p03_runner import temporal_diagnostics
from .runner import gpu_lock
from utils.image_utils import psnr


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_or_validate_bundle(adapter, directory: Path, scene, camera_name: str):
    if directory.exists():
        restored = load_bundle(directory, adapter.args)
        if not (np.array_equal(restored.static_rows, adapter.static_rows) and np.array_equal(restored.dynamic_rows, adapter.dynamic_rows)):
            raise RuntimeError(f"Existing bundle stable IDs differ: {directory}")
    else:
        adapter.save_bundle(directory)
        restored = load_bundle(directory, adapter.args)
    check = reload_check(adapter, restored, scene, camera_name)
    if not check["passed"]:
        raise RuntimeError(check)
    return restored, check


def ensure_consensus_gt(scene, camera_names: list[str], times: list[int], max_attempts: int = 5) -> dict:
    times = sorted(set(int(value) for value in times))
    report = {"schema": "p10-extra-consensus-gt-v1", "required_matches": 2, "max_attempts": max_attempts, "cameras": {}}
    for name in camera_names:
        base = select_camera(scene, name, times[0])
        source = Path(base.image_path)
        if source.suffix.lower() != ".mp4":
            source = source.parents[1] / f"{name}.mp4"
        missing = [timestamp for timestamp in times if not (VERIFIED_GT_ROOT / name / f"{timestamp:06d}.png").exists()]
        counts = {}
        accepted = {}
        attempts_used = 0
        for attempt in range(max_attempts):
            if not missing:
                break
            decoder = VideoFrameDecoder(source)
            try:
                for timestamp in times:
                    image = decoder.read(timestamp)
                    if timestamp not in missing:
                        continue
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    key = (timestamp, digest)
                    counts[key] = counts.get(key, 0) + 1
                    if counts[key] >= 2 and timestamp not in accepted:
                        accepted[timestamp] = (digest, image.copy())
            finally:
                decoder.close()
            attempts_used = attempt + 1
            missing = [timestamp for timestamp in missing if timestamp not in accepted]
        if missing:
            raise RuntimeError(f"No two matching P10 H.264 decodes for {name}: {missing}")
        directory = VERIFIED_GT_ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        for timestamp, (_, image) in accepted.items():
            destination = directory / f"{timestamp:06d}.png"
            temporary = directory / f".{timestamp:06d}.p10.tmp"
            image.save(temporary, format="PNG")
            temporary.replace(destination)
        report["cameras"][name] = {
            "source": str(source),
            "times": times,
            "new_frames": len(accepted),
            "attempts_used": attempts_used,
            "sha256": {str(timestamp): digest for timestamp, (digest, _) in accepted.items()},
        }
    dump(VERIFIED_GT_ROOT / "p10_extra_consensus.json", report)
    return report


@torch.no_grad()
def event_patch_diagnostics(reference, r0, r1, scene, camera_names: list[str], times: list[int], demands: list[dict]) -> dict:
    unique = {}
    for demand in demands:
        point = int(demand["global_index"])
        sample_index = int(demand["reappearance_sample"])
        unique[(point, sample_index)] = demand["repair_status"]
    by_sample = {}
    for (point, sample_index), status in unique.items():
        by_sample.setdefault(sample_index, []).append((point, status))
    rows = []
    decoders = {}
    try:
        for sample_index, events in sorted(by_sample.items()):
            timestamp = int(times[sample_index])
            xyz = reference.model.get_xyz_at_t(timestamp, training=False)
            for name in camera_names:
                camera = select_camera(scene, name, timestamp)
                gt = load_ground_truth(camera, decoders)
                images = {"R0": render_one(r0, camera), "R1": render_one(r1, camera), "reference": render_one(reference, camera)}
                height, width = int(gt.shape[1]), int(gt.shape[2])
                for point, status in events:
                    center = project_centers(xyz[point : point + 1], camera.full_proj_transform, width, height)[0]
                    cx, cy = float(center[0]), float(center[1])
                    radius = 32.0
                    x0 = max(0, int(np.floor(cx - radius)))
                    x1 = min(width, int(np.ceil(cx + radius + 1)))
                    y0 = max(0, int(np.floor(cy - radius)))
                    y1 = min(height, int(np.ceil(cy + radius + 1)))
                    if x0 >= x1 or y0 >= y1:
                        rows.append({"global_index": point, "sample_index": sample_index, "timestamp": timestamp, "camera": name, "repair_status": status, "visible_patch": False, "center_x": cx, "center_y": cy})
                        continue
                    yy, xx = torch.meshgrid(torch.arange(y0, y1, device="cuda"), torch.arange(x0, x1, device="cuda"), indexing="ij")
                    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
                    if not bool(mask.any()):
                        continue
                    row = {"global_index": point, "sample_index": sample_index, "timestamp": timestamp, "camera": name, "repair_status": status, "visible_patch": True, "center_x": cx, "center_y": cy, "pixels": int(mask.sum())}
                    for label, image in images.items():
                        residual = (image[:, y0:y1, x0:x1] - gt[:, y0:y1, x0:x1]) ** 2
                        row[f"{label}_gt_mse"] = float(residual[:, mask].mean())
                    row["R1_minus_R0_gt_mse"] = row["R1_gt_mse"] - row["R0_gt_mse"]
                    rows.append(row)
    finally:
        for decoder in decoders.values():
            decoder.close()
    visible = [row for row in rows if row.get("visible_patch")]
    repaired = [row for row in visible if row["repair_status"] == "repaired"]
    def summarize(group):
        return {
            "rows": len(group),
            "mean_R0_gt_mse": statistics.fmean(row["R0_gt_mse"] for row in group) if group else None,
            "mean_R1_gt_mse": statistics.fmean(row["R1_gt_mse"] for row in group) if group else None,
            "mean_R1_minus_R0_gt_mse": statistics.fmean(row["R1_minus_R0_gt_mse"] for row in group) if group else None,
        }
    return {"patch_radius_full_resolution_pixels": 32, "rows": rows, "all_visible": summarize(visible), "repaired_visible": summarize(repaired)}


@torch.no_grad()
def extra_reappearance_diagnostics(reference, r0, r1, scene, camera_names: list[str], demands: list[dict], windows: list[list[int]], times: list[int], output: Path) -> dict:
    covered = {timestamp for start, end in windows for timestamp in range(start, end + 1)}
    reappearance_times = sorted({int(times[int(row["reappearance_sample"])]) for row in demands})
    centers = [timestamp for timestamp in reappearance_times if timestamp not in covered][:5]
    if not centers:
        result = {"status": "COVERED_BY_W", "centers": [], "times": [], "rows": []}
        dump(output / "extra_reappearance.json", result)
        return result
    diagnostic_times = sorted({value for center in centers for value in range(max(0, center - 2), min(299, center + 2) + 1)})
    consensus = ensure_consensus_gt(scene, camera_names, diagnostic_times)
    rows = []
    decoders = {}
    try:
        for name in camera_names:
            for timestamp in diagnostic_times:
                camera = select_camera(scene, name, timestamp)
                gt = load_ground_truth(camera, decoders)
                for label, model in (("R0", r0), ("R1", r1), ("reference", reference)):
                    image = render_one(model, camera)
                    rows.append({"camera": name, "timestamp": timestamp, "method": label, "gt_mse": float(((image - gt) ** 2).mean()), "psnr": float(psnr(image[None], gt[None]).item())})
    finally:
        for decoder in decoders.values():
            decoder.close()
    with (output / "extra_reappearance_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "COMPLETED", "centers": centers, "times": diagnostic_times, "image_count_per_method": len(camera_names) * len(diagnostic_times), "consensus": consensus, "rows": rows}
    dump(output / "extra_reappearance.json", result)
    return result


def main() -> None:
    config_path = ROOT / "configs" / "research" / "p10.json"
    splits_path = ROOT / "research" / "manifests" / "splits.json"
    samples_path = ROOT / "research" / "manifests" / "samples.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P10" / "cut_roasted_beef" / "b0.5"
    base.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        cache_root = ROOT / "research_cache" / "cut_roasted_beef"
        k1_meta = build_k1(reference, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], cache_root)
        k2_meta = build_k2(reference, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], cache_root)
        k4_meta = build_k4(reference, samples["T24"], cache_root)
        k1 = torch.load(k1_meta["path"], map_location="cpu")
        k2 = torch.load(k2_meta["path"], map_location="cpu")
        k4 = torch.load(k4_meta["path"], map_location="cpu")
        checks = correctness_checks()
        if not checks["passed"]:
            raise RuntimeError(checks)
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        selection_started = time.perf_counter()
        all_demands = enumerate_reappearing_demands(k1["s_it"].double().numpy(), k2["e_it"].double().numpy(), reference.static_count)
        demand_limit = min(config["maximum_demands"], int(np.floor(0.1 * 124447)))
        demands = all_demands[:demand_limit]
        feasibility = attach_feasible_candidates(reference, scene, demands, k1["s_it"].double().numpy(), k4, splits["c4"], samples["T24"], samples["statistics_resolution"], select_camera)
        probe = {
            "all_demands": len(all_demands),
            "unique_reappearing_points": len({row["global_index"] for row in all_demands}),
            "selected_demands": len(demands),
            "selected_unique_points": len({row["global_index"] for row in demands}),
            "static_selected_demands": sum(row["kind"] == "static" for row in demands),
            "dynamic_selected_demands": sum(row["kind"] == "dynamic" for row in demands),
            **feasibility,
        }
        dump(base / "event_probe.json", probe)
        if feasibility["feasible_demands"] < config["minimum_feasible_demands"]:
            dump(base / "status.json", {"status": "BLOCKED_EVENT_INSUFFICIENT", **probe})
            print(json.dumps({"status": "BLOCKED_EVENT_INSUFFICIENT", **probe}, indent=2))
            return
        p02 = load_bundle(ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0" / "bundle", reference.args)
        scores = k2["e_it"].double().numpy().mean(axis=0)
        r0_selection = repair_selection(reference, p02.static_rows, p02.dynamic_rows, demands, scores)
        repeats = [repair_selection(reference, p02.static_rows, p02.dynamic_rows, copy.deepcopy(demands), scores) for _ in (0, 1, 2)]
        checks["repair_seed_invariant_0_1_2"] = bool(all(np.array_equal(r0_selection.static_indices, item.static_indices) and np.array_equal(r0_selection.dynamic_indices, item.dynamic_indices) for item in repeats))
        checks["repair_keeps_kind_counts"] = r0_selection.diagnostics["static_count"] == p02.static_count and r0_selection.diagnostics["dynamic_count"] == p02.dynamic_count
        checks["repair_constraints_monotone"] = bool(r0_selection.diagnostics["all_processed_constraints_satisfied"])
        checks["r1_is_p02"] = True
        checks["passed"] = bool(all(value for value in checks.values() if isinstance(value, bool)))
        if not checks["passed"]:
            raise RuntimeError(checks)
        selection_seconds = time.perf_counter() - selection_started
        models = {
            "R0": reference.gather(r0_selection.static_indices, r0_selection.dynamic_indices),
            "R1": p02,
        }
        results = {}
        for rule, model in models.items():
            output = base / rule
            output.mkdir(parents=True, exist_ok=True)
            model, roundtrip = save_or_validate_bundle(model, output / "bundle", scene, splits["development"][0])
            models[rule] = model
            smoke = smoke_render(model, scene, [splits["development"][0]], [0, 149], output / "smoke")
            summary = evaluate_against_reference(model, reference, scene, splits["development"], samples["T24"], output)
            temporal = temporal_diagnostics(model, reference, scene, splits["development"], samples["windows"])
            dump(output / "temporal_diagnostics.json", {key: value for key, value in temporal.items() if key != "pairs"})
            inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
            latency = benchmark_latency(model, scene, inputs, output / "latency.json")
            save_comparison_visualizations(model, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
            selection_rows = r0_selection.static_indices if rule == "R0" else p02.static_rows
            selection_dynamic = r0_selection.dynamic_indices if rule == "R0" else p02.dynamic_rows
            np.savez_compressed(output / "kept_ids.npz", static_rows=selection_rows, dynamic_rows=selection_dynamic)
            results[rule] = {"rule": rule, **summary, "temporal": {key: value for key, value in temporal.items() if key != "pairs"}, "roundtrip": roundtrip, "resources": {"bundle_bytes": directory_bytes(output / "bundle"), "actual_total_points": model.total_count, "static_points": model.static_count, "dynamic_points": model.dynamic_count, "smoke": smoke, "latency": latency}}
        event_diagnostics = event_patch_diagnostics(reference, models["R0"], models["R1"], scene, splits["development"], samples["T24"], r0_selection.demands)
        dump(base / "event_patch_diagnostics.json", event_diagnostics)
        extra = extra_reappearance_diagnostics(reference, models["R0"], models["R1"], scene, splits["development"], r0_selection.demands, samples["windows"], samples["T24"], base)
    demand_report = r0_selection.demands
    dump(base / "demands.json", demand_report)
    dump(base / "repair.json", r0_selection.diagnostics)
    dump(base / "correctness_checks.json", checks)
    for rule, result in results.items():
        output = base / rule
        dump(output / "resolved_config.json", {**config, "rule": rule, "k1_key": k1_meta["key"], "k2_key": k2_meta["key"], "k4_key": k4_meta["key"]})
        dump(output / "resources.json", result["resources"])
        dump(output / "timing.json", {"shared_selection_seconds": selection_seconds})
        dump(output / "correctness_checks.json", checks)
        dump(output / "provenance.json", {"git": git_provenance(), "checkpoint_sha256": reference.checkpoint_sha256, "k1_key": k1_meta["key"], "k2_key": k2_meta["key"], "k4_key": k4_meta["key"], "config_sha256": sha256_file(config_path), "splits_manifest_sha256": sha256_file(splits_path), "samples_manifest_sha256": sha256_file(samples_path)})
        (output / "commands.txt").write_text("python -m research_impl.p10_runner\n", encoding="utf-8")
        dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
    aggregate = {"status": "COMPLETED", "probe": probe, "correctness_checks": checks, "repair": r0_selection.diagnostics, "event_patch_summary": {key: value for key, value in event_diagnostics.items() if key != "rows"}, "extra_reappearance_summary": {key: value for key, value in extra.items() if key not in {"rows", "consensus"}}, "runs": [results[rule] for rule in config["rules"]]}
    dump(base.parent / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
