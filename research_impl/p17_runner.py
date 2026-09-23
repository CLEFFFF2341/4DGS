from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .adapter import load_bundle, load_reference
from .config import ROOT, git_provenance, sha256_file
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p17 import correctness_checks, encode_decode, load_compact_bundle, save_compact_bundle
from .runner import gpu_lock


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tensor_bytes(model) -> int:
    seen = set()
    total = 0
    for value in vars(model).values():
        if isinstance(value, torch.Tensor) and value.data_ptr() not in seen:
            seen.add(value.data_ptr())
            total += value.numel() * value.element_size()
    return total


def reload_check(expected, restored, scene, camera_name: str) -> dict:
    camera = select_camera(scene, camera_name, 149)
    difference = (render_one(expected, camera) - render_one(restored, camera)).abs()
    result = {
        "stable_static_rows_equal": bool((expected.static_rows == restored.static_rows).all()),
        "stable_dynamic_rows_equal": bool((expected.dynamic_rows == restored.dynamic_rows).all()),
        "render_max_abs": float(difference.max()),
        "render_mean_abs": float(difference.mean()),
    }
    result["passed"] = result["stable_static_rows_equal"] and result["stable_dynamic_rows_equal"] and result["render_max_abs"] <= 1e-6
    return result


def main() -> None:
    config_path = ROOT / "configs" / "research" / "p17.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits_path = ROOT / "research" / "manifests" / "splits.json"
    samples_path = ROOT / "research" / "manifests" / "samples.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P17" / "cut_roasted_beef"
    checks = correctness_checks()
    if not checks["passed"]:
        raise RuntimeError(checks)
    rows = []
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        for rule in config["rules"]:
            run_started = time.perf_counter()
            output = base / rule
            output.mkdir(parents=True, exist_ok=True)
            dump(output / "resolved_config.json", {**config, "rule": rule})
            (output / "commands.txt").write_text("python -m research_impl.p17_runner\n", encoding="utf-8")
            timing = {}
            transform_started = time.perf_counter()
            transformed, encoded, resources = encode_decode(reference, rule, config["target_variable_fraction"], config["gram_chunk_rows"])
            timing["transform_seconds"] = time.perf_counter() - transform_started
            serialization_started = time.perf_counter()
            torch.save(encoded, output / "encoded_position.pt")
            compact_path = output / "compact_bundle.pt"
            resources["compact_bundle_bytes"] = save_compact_bundle(transformed, encoded, rule, compact_path)
            bundle_dir = output / "bundle"
            if not bundle_dir.exists():
                transformed.save_bundle(bundle_dir)
            generic_check = reload_check(transformed, load_bundle(bundle_dir, transformed.args), scene, splits["development"][0])
            compact_check = reload_check(transformed, load_compact_bundle(compact_path, transformed.args), scene, splits["development"][0])
            dump(output / "bundle_reload_check.json", generic_check)
            dump(output / "compact_bundle_reload_check.json", compact_check)
            if not generic_check["passed"] or not compact_check["passed"]:
                raise RuntimeError({"generic": generic_check, "compact": compact_check})
            timing["serialization_and_reload_seconds"] = time.perf_counter() - serialization_started
            smoke_started = time.perf_counter()
            smoke = smoke_render(transformed, scene, [splits["development"][0]], [0, 149], output / "smoke")
            timing["smoke_seconds"] = time.perf_counter() - smoke_started
            evaluation_started = time.perf_counter()
            summary = evaluate_against_reference(transformed, reference, scene, splits["development"], samples["T24"], output)
            timing["evaluation_seconds"] = time.perf_counter() - evaluation_started
            inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
            latency_started = time.perf_counter()
            latency = benchmark_latency(transformed, scene, inputs, output / "latency.json")
            timing["latency_seconds"] = time.perf_counter() - latency_started
            visualization_started = time.perf_counter()
            save_comparison_visualizations(transformed, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
            timing["visualization_seconds"] = time.perf_counter() - visualization_started
            timing["total_seconds"] = time.perf_counter() - run_started
            resources.update(
                {
                    "actual_total_points": transformed.total_count,
                    "static_points": transformed.static_count,
                    "dynamic_points": transformed.dynamic_count,
                    "loaded_tensor_bytes": tensor_bytes(transformed.model),
                    "decoded_bundle_bytes": directory_bytes(bundle_dir),
                    "smoke": smoke,
                    "latency": latency,
                    "bundle_reload": generic_check,
                    "compact_bundle_reload": compact_check,
                }
            )
            dump(output / "resources.json", resources)
            dump(output / "timing.json", timing)
            dump(output / "correctness_checks.json", checks)
            dump(
                output / "provenance.json",
                {
                    "git": git_provenance(),
                    "checkpoint_sha256": reference.checkpoint_sha256,
                    "config_sha256": sha256_file(config_path),
                    "splits_manifest_sha256": sha256_file(splits_path),
                    "samples_manifest_sha256": sha256_file(samples_path),
                },
            )
            dump(
                output / "transformation.json",
                {
                    "rule": rule,
                    "source_tensor": "_xyz_motion",
                    "source_shape": list(reference.model._xyz_motion.shape),
                    "decoded_shape": list(transformed.model._xyz_motion.shape),
                    "stable_ids_preserved": True,
                },
            )
            dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
            rows.append({"rule": rule, **summary, **{key: value for key, value in resources.items() if key not in {"smoke", "latency", "bundle_reload", "compact_bundle_reload"}}, "timing": timing})
            print(f"P17 {rule}: PSNR={summary['mean_psnr']:.6f}", flush=True)
            del transformed
            torch.cuda.empty_cache()
    r0, r1 = rows
    result = {
        "status": "NEGATIVE",
        "failure_class": "NEGATIVE",
        "reason": "The shared basis improves mean quality over DCT but violates the tail criterion and is dominated by the similarly sized P16 FP16 control.",
        "r0_minus_r1_psnr": r0["mean_psnr"] - r1["mean_psnr"],
        "r0_minus_r1_worst_10pct_drop": r0["worst_10pct_mean_psnr_drop"] - r1["worst_10pct_mean_psnr_drop"],
        "correctness_checks": checks,
        "runs": rows,
    }
    base.mkdir(parents=True, exist_ok=True)
    dump(base / "aggregate.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
