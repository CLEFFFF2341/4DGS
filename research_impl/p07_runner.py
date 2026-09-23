from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .adapter import load_bundle, load_reference
from .cache import build_k2
from .config import ROOT, git_provenance, sha256_file
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p01 import quotas
from .methods.p07 import common_id_diagnostics, correctness_checks, round_state, round_target, score_maps, select_round
from .runner import gpu_lock


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cache_metadata(meta: dict) -> dict:
    path = Path(meta["path"])
    metadata_path = path.parent / "metadata.json"
    stored = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {**stored, **meta}


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
            raise RuntimeError(f"Existing bundle has different stable IDs: {directory}")
    else:
        adapter.save_bundle(directory)
        restored = load_bundle(directory, adapter.args)
    check = reload_check(adapter, restored, scene, camera_name)
    if not check["passed"]:
        raise RuntimeError(check)
    return restored, check


def current_original_rho(adapter, scores: np.ndarray, original_static: dict[int, float], original_dynamic: dict[int, float]) -> float:
    original = np.concatenate(
        (
            np.asarray([original_static[int(row)] for row in adapter.static_rows], dtype=np.float64),
            np.asarray([original_dynamic[int(row)] for row in adapter.dynamic_rows], dtype=np.float64),
        )
    )
    return float(spearmanr(original, scores).statistic)


def run_two_frame_smoke(reference, scene, config: dict, splits: dict, samples: dict, base: Path) -> dict:
    output = base / "smoke_2frame"
    output.mkdir(parents=True, exist_ok=True)
    smoke_times = [0, 149]
    rows = {}
    for rule in config["rules"]:
        current = reference
        _, final_static, final_dynamic = quotas(reference.static_count, reference.dynamic_count, config["budget_fraction"])
        original_static_scores = None
        original_dynamic_scores = None
        round_rows = []
        for round_index in range(1, int(config["rounds"]) + 1):
            request_started = time.perf_counter()
            meta = build_k2(
                current,
                scene,
                [splits["c4"][0]],
                smoke_times,
                samples["statistics_resolution"],
                ROOT / "research_cache" / "cut_roasted_beef",
            )
            request_seconds = time.perf_counter() - request_started
            cache = torch.load(meta["path"], map_location="cpu")
            scores = cache["e_it"].double().numpy().mean(axis=0)
            if original_static_scores is None:
                original_static_scores, original_dynamic_scores = score_maps(current, scores)
            selection = select_round(
                current,
                cache,
                round_target(reference.static_count, final_static, round_index, int(config["rounds"])),
                round_target(reference.dynamic_count, final_dynamic, round_index, int(config["rounds"])),
                original_static_scores if rule == "R1" else None,
                original_dynamic_scores if rule == "R1" else None,
            )
            metadata = cache_metadata(meta)
            round_rows.append(
                {
                    "round": round_index,
                    "points_before": current.total_count,
                    "points_after": int(selection.static_indices.size + selection.dynamic_indices.size),
                    "k2_key": meta["key"],
                    "parent_model_sha256": cache["specification"]["parent_model_sha256"],
                    "request_seconds": request_seconds,
                    "measured_build_seconds": float(metadata.get("seconds", request_seconds)),
                    "mean_frame_seconds": float(metadata.get("mean_frame_seconds", 0.0)),
                    "peak_allocated_bytes": int(metadata.get("peak_allocated_bytes", 0)),
                    "peak_reserved_bytes": int(metadata.get("peak_reserved_bytes", 0)),
                    "replay_max_abs": float(cache["replay_max_abs"]),
                }
            )
            current = current.gather(selection.static_indices, selection.dynamic_indices)
            torch.cuda.empty_cache()
        estimated_full_statistics_seconds = float(sum(row["mean_frame_seconds"] * len(splits["c4"]) * len(samples["T24"]) for row in round_rows))
        rows[rule] = {
            "rounds": round_rows,
            "final_static": current.static_count,
            "final_dynamic": current.dynamic_count,
            "estimated_full_statistics_seconds": estimated_full_statistics_seconds,
            "within_45_minute_limit": estimated_full_statistics_seconds <= 45 * 60,
        }
    result = {
        "status": "PASSED" if all(row["within_45_minute_limit"] for row in rows.values()) else "BLOCKED_BUDGET",
        "cameras": [splits["c4"][0]],
        "times": smoke_times,
        "rows": rows,
    }
    dump(output / "smoke.json", result)
    if result["status"] != "PASSED":
        raise RuntimeError(result)
    return result


def run_rule(rule: str, reference, scene, config: dict, splits: dict, samples: dict, base: Path) -> tuple[object, dict]:
    output = base / rule
    output.mkdir(parents=True, exist_ok=True)
    current = reference
    rounds = int(config["rounds"])
    _, final_static, final_dynamic = quotas(reference.static_count, reference.dynamic_count, config["budget_fraction"])
    previous = None
    original_static_scores = None
    original_dynamic_scores = None
    records = []
    rule_started = time.perf_counter()
    for round_index in range(1, rounds + 1):
        round_started = time.perf_counter()
        round_dir = output / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        k2_started = time.perf_counter()
        k2_meta = build_k2(
            current,
            scene,
            splits["c4"],
            samples["T24"],
            samples["statistics_resolution"],
            ROOT / "research_cache" / "cut_roasted_beef",
        )
        k2_request_seconds = time.perf_counter() - k2_started
        k2 = torch.load(k2_meta["path"], map_location="cpu")
        fresh_scores = k2["e_it"].double().numpy().mean(axis=0)
        hits = k2["hit_counts"].long().numpy()
        if original_static_scores is None:
            original_static_scores, original_dynamic_scores = score_maps(current, fresh_scores)
        target_static = round_target(reference.static_count, final_static, round_index, rounds)
        target_dynamic = round_target(reference.dynamic_count, final_dynamic, round_index, rounds)
        selection = select_round(
            current,
            k2,
            target_static,
            target_dynamic,
            original_static_scores if rule == "R1" else None,
            original_dynamic_scores if rule == "R1" else None,
        )
        repeats = [
            select_round(
                current,
                k2,
                target_static,
                target_dynamic,
                original_static_scores if rule == "R1" else None,
                original_dynamic_scores if rule == "R1" else None,
            )
            for _ in (0, 1, 2)
        ]
        decision_repeat_invariant = bool(
            all(np.array_equal(selection.static_indices, item.static_indices) for item in repeats)
            and all(np.array_equal(selection.dynamic_indices, item.dynamic_indices) for item in repeats)
        )
        exposure = common_id_diagnostics(previous, current, fresh_scores, hits)
        fresh_vs_original = current_original_rho(current, fresh_scores, original_static_scores, original_dynamic_scores)
        parent_static_rows = current.static_rows.copy()
        parent_dynamic_rows = current.dynamic_rows.copy()
        next_model = current.gather(selection.static_indices, selection.dynamic_indices)
        restored, roundtrip = save_or_validate_bundle(next_model, round_dir / "bundle", scene, splits["development"][0])
        np.savez_compressed(
            round_dir / "kept_ids.npz",
            static_indices=selection.static_indices,
            dynamic_indices=selection.dynamic_indices,
            static_rows=next_model.static_rows,
            dynamic_rows=next_model.dynamic_rows,
            parent_static_rows=parent_static_rows,
            parent_dynamic_rows=parent_dynamic_rows,
        )
        torch.save(
            {
                "fresh_scores": torch.from_numpy(selection.fresh_scores),
                "decision_scores": torch.from_numpy(selection.decision_scores),
                "hit_counts": torch.from_numpy(hits),
            },
            round_dir / "scores.pt",
        )
        metadata = cache_metadata(k2_meta)
        record = {
            "round": round_index,
            "k2_key": k2_meta["key"],
            "parent_model_sha256": k2["specification"]["parent_model_sha256"],
            "cache_reused": bool(k2_meta.get("reused", False)),
            "cache_build_seconds": float(metadata.get("seconds", 0.0)),
            "cache_request_seconds": k2_request_seconds,
            "cache_bytes": int(Path(k2_meta["path"]).stat().st_size),
            "replay_max_abs": float(k2["replay_max_abs"]),
            "finite_scores": bool(np.isfinite(fresh_scores).all()),
            "decision_repeat_invariant": decision_repeat_invariant,
            "fresh_vs_original_spearman": fresh_vs_original,
            "selection": selection.diagnostics,
            "visibility_change": exposure,
            "roundtrip": roundtrip,
            "bundle_bytes": directory_bytes(round_dir / "bundle"),
            "round_wall_seconds": time.perf_counter() - round_started,
        }
        dump(round_dir / "round.json", record)
        records.append(record)
        previous = round_state(current, fresh_scores, hits)
        current = restored
        print(
            f"P07 {rule} round {round_index}/{rounds}: {current.total_count} points, "
            f"rho(prev)={exposure['fresh_score_spearman']}, new_rays={exposure['newly_exposed_ray_count']}",
            flush=True,
        )
    final_model, final_roundtrip = save_or_validate_bundle(current, output / "bundle", scene, splits["development"][0])
    smoke = smoke_render(final_model, scene, [splits["development"][0]], [0, 149], output / "smoke")
    summary = evaluate_against_reference(final_model, reference, scene, splits["development"], samples["T24"], output)
    inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
    latency = benchmark_latency(final_model, scene, inputs, output / "latency.json")
    save_comparison_visualizations(final_model, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
    np.savez_compressed(output / "kept_ids.npz", static_rows=final_model.static_rows, dynamic_rows=final_model.dynamic_rows)
    result = {
        "rule": rule,
        **summary,
        "rounds": records,
        "final_roundtrip": final_roundtrip,
        "total_wall_seconds": time.perf_counter() - rule_started,
        "resources": {
            "actual_total_points": final_model.total_count,
            "static_points": final_model.static_count,
            "dynamic_points": final_model.dynamic_count,
            "bundle_bytes": directory_bytes(output / "bundle"),
            "logical_k2_builds": rounds,
            "logical_k2_build_seconds": float(sum(row["cache_build_seconds"] for row in records)),
            "actual_k2_request_seconds": float(sum(row["cache_request_seconds"] for row in records)),
            "smoke": smoke,
            "latency": latency,
        },
    }
    dump(output / "rounds.json", records)
    dump(output / "summary_with_rounds.json", result)
    return final_model, result


def main() -> None:
    config_path = ROOT / "configs" / "research" / "p07.json"
    splits_path = ROOT / "research" / "manifests" / "splits.json"
    samples_path = ROOT / "research" / "manifests" / "samples.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P07" / "cut_roasted_beef" / "b0.5"
    base.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        smoke = run_two_frame_smoke(reference, scene, config, splits, samples, base)
        models = {}
        results = {}
        for rule in config["rules"]:
            models[rule], results[rule] = run_rule(rule, reference, scene, config, splits, samples, base)
        p02 = load_bundle(ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0" / "bundle", reference.args)
        round_records = {rule: results[rule]["rounds"] for rule in results}
        checks = correctness_checks(reference, models["R0"], models["R1"], p02.static_rows, p02.dynamic_rows, round_records)
        p00_adapter = json.loads((ROOT / "runs" / "research" / "P00" / "adapter_checks.json").read_text(encoding="utf-8"))
        p00_retention = json.loads((ROOT / "runs" / "research" / "P00" / "retention_check.json").read_text(encoding="utf-8"))
        checks["s2_true_means_delete_inherited"] = bool(p00_adapter["true_means_delete"])
        checks["s2_tensor_gather_inherited"] = bool(p00_adapter["all_tensor_gathers_passed"])
        checks["s2_empty_kinds_inherited"] = bool(p00_adapter["empty_kind"]["dynamic_only_shape"] and p00_adapter["empty_kind"]["static_only_shape"])
        checks["s2_full_retain_max_abs"] = float(p00_retention["max_abs"])
        checks["passed"] = bool(checks["passed"] and checks["s2_true_means_delete_inherited"] and checks["s2_tensor_gather_inherited"] and checks["s2_empty_kinds_inherited"] and checks["s2_full_retain_max_abs"] <= 1e-6)
        if not checks["passed"]:
            raise RuntimeError(checks)
    for rule, result in results.items():
        output = base / rule
        dump(output / "resolved_config.json", {**config, "rule": rule})
        dump(output / "correctness_checks.json", checks)
        dump(output / "resources.json", result["resources"])
        dump(output / "timing.json", {"total_wall_seconds": result["total_wall_seconds"], "round_wall_seconds": [row["round_wall_seconds"] for row in result["rounds"]]})
        dump(
            output / "provenance.json",
            {
                "git": git_provenance(),
                "checkpoint_sha256": reference.checkpoint_sha256,
                "round_k2_keys": [row["k2_key"] for row in result["rounds"]],
                "round_parent_model_sha256": [row["parent_model_sha256"] for row in result["rounds"]],
                "config_sha256": sha256_file(config_path),
                "splits_manifest_sha256": sha256_file(splits_path),
                "samples_manifest_sha256": sha256_file(samples_path),
            },
        )
        (output / "commands.txt").write_text("python -m research_impl.p07_runner\n", encoding="utf-8")
        dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
    aggregate = {"status": "COMPLETED", "smoke": smoke, "correctness_checks": checks, "runs": [results[rule] for rule in config["rules"]]}
    dump(base.parent / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
