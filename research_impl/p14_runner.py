from __future__ import annotations

import json
from pathlib import Path

import torch

from .adapter import load_reference
from .config import ROOT, git_provenance
from .evaluate import benchmark_latency, build_verified_ground_truth_cache, evaluate_against_reference, save_comparison_visualizations, smoke_render
from .methods.p14 import convert_candidates
from .runner import gpu_lock


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    output = ROOT / "runs" / "research" / "P14" / "cut_roasted_beef" / "reachable"
    output.mkdir(parents=True, exist_ok=True)
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    probe_summary = json.loads((ROOT / "runs" / "research" / "P14" / "probe.json").read_text(encoding="utf-8"))
    probe = torch.load(ROOT / "runs" / "research" / "P14" / "probe.pt", map_location="cpu")
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        build_verified_ground_truth_cache(reference, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        transformed, mapping = convert_candidates(reference, probe)
        bundle_dir = output / "bundle"
        if not bundle_dir.exists():
            bundle = transformed.save_bundle(bundle_dir)
        else:
            bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
        smoke = smoke_render(transformed, scene, [splits["development"][0]], [0, 149], output / "smoke")
        summary = evaluate_against_reference(transformed, reference, scene, splits["development"], samples["T24"], output)
        inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
        latency = benchmark_latency(transformed, scene, inputs, output / "latency.json")
        save_comparison_visualizations(transformed, reference, scene, splits["development"], samples["fixed_visualization_times"], output / "visualizations")
    write(output / "mapping.json", mapping)
    write(output / "resources.json", {"probe": probe_summary, "bundle": bundle, "smoke": smoke, "latency": latency})
    write(output / "provenance.json", {"git": git_provenance(), "checkpoint_sha256": reference.checkpoint_sha256})
    write(output / "status.json", {"status": "BLOCKED_TARGET", "target_reachable": False, "converted_count": len(mapping["converted_dynamic_rows"])})
    print(json.dumps({"status": "BLOCKED_TARGET", "converted_count": len(mapping["converted_dynamic_rows"]), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
