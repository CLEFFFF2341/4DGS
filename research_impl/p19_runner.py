from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaussian_renderer import render

from .adapter import load_bundle, load_reference
from .cache import build_k0, build_k1
from .config import ROOT, git_provenance, sha256_file
from .evaluate import (
    benchmark_latency,
    evaluate_against_reference,
    pipeline,
    render_one,
    save_comparison_visualizations,
    select_camera,
    smoke_render,
)
from .methods.p19 import (
    CostSelection,
    decision_toy,
    select_cost,
    selection_from_indices,
)
from .runner import gpu_lock


OUTPUT_ROOT = ROOT / "runs" / "research" / "P19" / "cut_roasted_beef"
P02_ROOT = ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0"
FULL_BUNDLE = ROOT / "runs" / "research" / "P18" / "cut_roasted_beef" / "active0.5" / "R0" / "bundle"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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
        result["static_rows_equal"]
        and result["dynamic_rows_equal"]
        and result["render_max_abs"] <= 1e-6
    )
    return result


def selections(
    reference: Any,
    scores: np.ndarray,
    tile_costs: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, CostSelection], dict[str, Any]]:
    full_bytes = directory_bytes(FULL_BUNDLE)
    byte_cap = int(np.floor(float(config["byte_cap_fraction"]) * full_bytes))
    static_cost = int(config["static_payload_bytes"]) + int(config["stable_id_bytes_per_point"])
    dynamic_cost = int(config["dynamic_payload_bytes"]) + int(config["stable_id_bytes_per_point"])
    byte_costs = np.concatenate(
        (
            np.full(reference.static_count, static_cost, dtype=np.float64),
            np.full(reference.dynamic_count, dynamic_cost, dtype=np.float64),
        )
    )
    header_upper_bound = full_bytes - int(byte_costs.sum())
    tile_cap = 0.5 * float(tile_costs.sum())
    p02 = np.load(P02_ROOT / "kept_ids.npz")
    result = {
        "R0": select_cost(
            reference,
            scores,
            byte_costs,
            byte_cap,
            "R0",
            mode="ratio",
            header_cost=header_upper_bound,
        ),
        "R1": select_cost(reference, scores, tile_costs, tile_cap, "R1", mode="ratio"),
        "R2": selection_from_indices(
            reference,
            scores,
            p02["static_indices"],
            p02["dynamic_indices"],
            "R2",
        ),
        "C0_byte_p02": select_cost(
            reference,
            scores,
            byte_costs,
            byte_cap,
            "C0_byte_p02",
            mode="score",
            header_cost=header_upper_bound,
        ),
        "C1_tile_p02": select_cost(
            reference,
            scores,
            tile_costs,
            tile_cap,
            "C1_tile_p02",
            mode="score",
        ),
    }
    caps = {
        "full_bundle_bytes": full_bytes,
        "byte_cap": byte_cap,
        "static_item_bytes": static_cost,
        "dynamic_item_bytes": dynamic_cost,
        "header_upper_bound_bytes": header_upper_bound,
        "full_tile_cost": float(tile_costs.sum()),
        "tile_cap": tile_cap,
    }
    return result, caps


@torch.no_grad()
def tile_forward_check(reference, scene, k1, splits, samples) -> dict[str, Any]:
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    stats_pipe = pipeline()
    stats_pipe.collect_stats = True
    measured = np.zeros(reference.total_count, dtype=np.int64)
    width, height = samples["statistics_resolution"]
    timestamp = samples["T24"][0]
    for name in splits["c4"]:
        camera = copy.copy(select_camera(scene, name, timestamp))
        camera.image_width = int(width)
        camera.image_height = int(height)
        stats = render(
            camera,
            reference.model,
            stats_pipe,
            background,
            timestamp=timestamp,
            near=reference.args.near,
            far=reference.args.far,
        )
        measured += stats["tiles_touched"].cpu().numpy().astype(np.int64)
    cached = k1["tile_touches"][0].numpy()
    return {
        "camera_count": len(splits["c4"]),
        "timestamp": timestamp,
        "maximum_absolute_error": int(np.max(np.abs(measured - cached))),
        "sum_measured": int(measured.sum()),
        "sum_cached": int(cached.sum()),
        "passed": bool(np.array_equal(measured, cached)),
    }


def correctness(reference, scene, scores, tile_costs, config, splits, samples, k1):
    selected, caps = selections(reference, scores, tile_costs, config)
    repeats = [selections(reference, scores, tile_costs, config)[0] for _ in range(2)]
    identity = reference.gather(range(reference.static_count), range(reference.dynamic_count))
    camera = select_camera(scene, "cam01", 149)
    identity_error = float((render_one(reference, camera) - render_one(identity, camera)).abs().max())
    p02 = np.load(P02_ROOT / "kept_ids.npz")
    checks = {
        "decision_toy": decision_toy(),
        "tile_forward_check": tile_forward_check(reference, scene, k1, splits, samples),
        "full_retention_identity_max_abs": identity_error,
        "full_retention_identity": identity_error <= 1e-6,
        "byte_rules_under_estimated_cap": all(
            selected[name].diagnostics["estimated_total_cost"] <= caps["byte_cap"]
            for name in ("R0", "C0_byte_p02")
        ),
        "tile_rules_under_cap": all(
            selected[name].diagnostics["selected_item_cost"] <= caps["tile_cap"]
            for name in ("R1", "C1_tile_p02")
        ),
        "r2_exactly_p02": bool(
            np.array_equal(selected["R2"].static_indices, p02["static_indices"])
            and np.array_equal(selected["R2"].dynamic_indices, p02["dynamic_indices"])
        ),
        "deterministic_seed_policy_0_1_2": all(
            np.array_equal(selected[name].static_indices, repeat[name].static_indices)
            and np.array_equal(selected[name].dynamic_indices, repeat[name].dynamic_indices)
            for repeat in repeats
            for name in selected
        ),
    }
    checks["passed"] = bool(
        checks["decision_toy"]["passed"]
        and checks["tile_forward_check"]["passed"]
        and all(value for value in checks.values() if isinstance(value, bool))
    )
    return selected, caps, checks


def run_selection(
    reference,
    scene,
    selection: CostSelection,
    scores,
    tile_costs,
    caps,
    config,
    splits,
    samples,
    cache_paths,
) -> dict[str, Any]:
    output = OUTPUT_ROOT / selection.name
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    subset = reference.gather(selection.static_indices, selection.dynamic_indices)
    np.savez_compressed(
        output / "kept_ids.npz",
        static_indices=selection.static_indices,
        dynamic_indices=selection.dynamic_indices,
        static_rows=subset.static_rows,
        dynamic_rows=subset.dynamic_rows,
    )
    combined = np.concatenate(
        (selection.static_indices, reference.static_count + selection.dynamic_indices)
    )
    diagnostics = {
        **selection.diagnostics,
        "selected_tile_cost": float(tile_costs[combined].sum()),
        "selected_e": float(scores[combined].sum()),
    }
    dump(output / "selection.json", diagnostics)
    bundle_dir = output / "bundle"
    bundle = subset.save_bundle(bundle_dir)
    actual_bytes = directory_bytes(bundle_dir)
    if selection.name in {"R0", "C0_byte_p02"} and actual_bytes > caps["byte_cap"]:
        raise RuntimeError(
            f"{selection.name} serialized bytes {actual_bytes} exceed cap {caps['byte_cap']}"
        )
    restored = load_bundle(bundle_dir, reference.args)
    roundtrip = reload_check(subset, restored, scene)
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
    resources = {
        "actual_total_points": subset.total_count,
        "static_points": subset.static_count,
        "dynamic_points": subset.dynamic_count,
        "bundle_bytes": actual_bytes,
        "byte_cap": caps["byte_cap"] if selection.name in {"R0", "C0_byte_p02"} else None,
        "selected_tile_cost": diagnostics["selected_tile_cost"],
        "tile_cap": caps["tile_cap"] if selection.name in {"R1", "C1_tile_p02"} else None,
        "smoke": smoke,
        "latency": latency,
    }
    dump(output / "resolved_config.json", {**config, "row": selection.name})
    dump(output / "bundle_reload_check.json", roundtrip)
    dump(output / "resources.json", resources)
    dump(output / "timing.json", {"total_seconds": time.perf_counter() - started})
    dump(
        output / "provenance.json",
        {
            "git": git_provenance(),
            "checkpoint_sha256": reference.checkpoint_sha256,
            "cache_sha256": {name: sha256_file(path) for name, path in cache_paths.items()},
            "config_sha256": sha256_file(ROOT / "configs" / "research" / "p19.json"),
        },
    )
    (output / "commands.txt").write_text(
        f"python -m research_impl.p19_runner  # {selection.name}\n", encoding="utf-8"
    )
    result = {
        "status": "COMPLETED",
        "row": selection.name,
        "selection": diagnostics,
        "summary": summary,
        "resources": resources,
        "bundle": bundle,
    }
    dump(output / "result.json", result)
    dump(output / "status.json", {"status": "COMPLETED", "row": selection.name})
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs" / "research" / "p19.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits_path = ROOT / "research" / "manifests" / "splits.json"
    samples_path = ROOT / "research" / "manifests" / "samples.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    scores = torch.load(P02_ROOT / "selection.pt", map_location="cpu")["scores"].numpy()
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        k0_meta = build_k0(reference, samples["T24"], ROOT / "research_cache" / "cut_roasted_beef")
        k1_meta = build_k1(
            reference,
            scene,
            splits["c4"],
            samples["T24"],
            samples["statistics_resolution"],
            ROOT / "research_cache" / "cut_roasted_beef",
        )
        k1 = torch.load(k1_meta["path"], map_location="cpu")
        tile_costs = k1["tile_touches"].double().sum(dim=0).numpy() / (
            len(splits["c4"]) * len(samples["T24"])
        )
        chosen, caps, checks = correctness(
            reference, scene, scores, tile_costs, config, splits, samples, k1
        )
        dump(OUTPUT_ROOT / "caps.json", caps)
        dump(OUTPUT_ROOT / "correctness_checks.json", checks)
        if not checks["passed"]:
            raise RuntimeError(checks)
        cache_paths = {
            "K0": Path(k0_meta["path"]),
            "K1": Path(k1_meta["path"]),
            "K2_scores": P02_ROOT / "selection.pt",
        }
        order = config["rules"] + config["controls"]
        results = {
            name: run_selection(
                reference,
                scene,
                chosen[name],
                scores,
                tile_costs,
                caps,
                config,
                splits,
                samples,
                cache_paths,
            )
            for name in order
        }
    baseline_latency_path = ROOT / "runs" / "research" / "P00" / "reference_latency.json"
    p12 = json.loads(
        (ROOT / "runs" / "research" / "P12" / "cut_roasted_beef" / "b0.5" / "aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = {
        "status": "COMPLETED",
        "seed_policy": "deterministic IDs verified for seeds 0,1,2; one quality evaluation per row",
        "caps": caps,
        "correctness": checks,
        "results": results,
        "p12": {
            "R0": p12["rules"]["R0"],
            "R1": p12["rules"]["R1"],
            "R2": p12["rules"]["R2"],
        },
        "reference_latency_path": str(baseline_latency_path),
    }
    for method, control in (("R0", "C0_byte_p02"), ("R1", "C1_tile_p02")):
        aggregate[f"{method}_minus_control_psnr"] = (
            results[method]["summary"]["mean_psnr"]
            - results[control]["summary"]["mean_psnr"]
        )
        aggregate[f"{method}_minus_control_lpips"] = (
            results[method]["summary"]["mean_lpips_alex"]
            - results[control]["summary"]["mean_lpips_alex"]
        )
    dump(OUTPUT_ROOT / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
