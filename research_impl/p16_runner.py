from __future__ import annotations

import json
from pathlib import Path

import torch

from .adapter import load_bundle, load_reference
from .config import ROOT, git_provenance
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, render_one, save_comparison_visualizations, select_camera, smoke_render
from .methods.p16 import correctness_checks, encode_decode, load_compact_bundle, save_compact_bundle
from .runner import gpu_lock


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P16" / "cut_roasted_beef"
    rows = []
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        checks = correctness_checks(reference)
        if not checks["passed"]:
            raise RuntimeError(checks)
        for rule in ("R0", "R1"):
            output = base / rule
            output.mkdir(parents=True, exist_ok=True)
            transformed, encoded, resources = encode_decode(reference, rule)
            torch.save(encoded, output / "encoded_position.pt")
            compact_path = output / "compact_bundle.pt"
            resources["compact_bundle_bytes"] = save_compact_bundle(transformed, encoded, rule, compact_path)
            bundle_dir = output / "bundle"
            if not bundle_dir.exists():
                transformed.save_bundle(bundle_dir)
            reloaded = load_bundle(bundle_dir, transformed.args)
            reload_camera = select_camera(scene, splits["development"][0], 149)
            reload_difference = (render_one(transformed, reload_camera) - render_one(reloaded, reload_camera)).abs()
            reload_check = {
                "stable_static_rows_equal": bool((transformed.static_rows == reloaded.static_rows).all()),
                "stable_dynamic_rows_equal": bool((transformed.dynamic_rows == reloaded.dynamic_rows).all()),
                "render_max_abs": float(reload_difference.max()),
                "render_mean_abs": float(reload_difference.mean()),
            }
            reload_check["passed"] = (
                reload_check["stable_static_rows_equal"]
                and reload_check["stable_dynamic_rows_equal"]
                and reload_check["render_max_abs"] <= 1e-6
            )
            dump(output / "bundle_reload_check.json", reload_check)
            if not reload_check["passed"]:
                raise RuntimeError(f"P16 {rule} bundle reload failed: {reload_check}")
            compact_reloaded = load_compact_bundle(compact_path, transformed.args)
            compact_difference = (render_one(transformed, reload_camera) - render_one(compact_reloaded, reload_camera)).abs()
            compact_reload_check = {
                "stable_static_rows_equal": bool((transformed.static_rows == compact_reloaded.static_rows).all()),
                "stable_dynamic_rows_equal": bool((transformed.dynamic_rows == compact_reloaded.dynamic_rows).all()),
                "render_max_abs": float(compact_difference.max()),
                "render_mean_abs": float(compact_difference.mean()),
            }
            compact_reload_check["passed"] = (
                compact_reload_check["stable_static_rows_equal"]
                and compact_reload_check["stable_dynamic_rows_equal"]
                and compact_reload_check["render_max_abs"] <= 1e-6
            )
            dump(output / "compact_bundle_reload_check.json", compact_reload_check)
            if not compact_reload_check["passed"]:
                raise RuntimeError(f"P16 {rule} compact bundle reload failed: {compact_reload_check}")
            smoke = smoke_render(transformed, scene, [splits["development"][0]], [0, 149], output / "smoke")
            summary = evaluate_against_reference(transformed, reference, scene, splits["development"], samples["T24"], output)
            inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
            latency = benchmark_latency(transformed, scene, inputs, output / "latency.json")
            save_comparison_visualizations(transformed, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
            dump(
                output / "resources.json",
                {**resources, "smoke": smoke, "latency": latency, "bundle_reload": reload_check, "compact_bundle_reload": compact_reload_check},
            )
            dump(output / "correctness_checks.json", checks)
            dump(output / "provenance.json", {"git": git_provenance(), "checkpoint_sha256": reference.checkpoint_sha256})
            dump(output / "status.json", {"status": "COMPLETED", "rule": rule})
            rows.append({"rule": rule, **summary, **resources})
            print(f"P16 {rule}: PSNR={summary['mean_psnr']:.6f}", flush=True)
    result = {"status": "COMPLETED", "correctness_checks": checks, "runs": rows}
    base.mkdir(parents=True, exist_ok=True)
    dump(base / "aggregate.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
