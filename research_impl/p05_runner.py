from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from . import p01_runner
from .adapter import load_reference
from .cache import build_k1, build_k4
from .config import ROOT
from .evaluate import build_verified_ground_truth_cache
from .methods import p05
from .runner import gpu_lock


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    config = json.loads((ROOT / "configs" / "research" / "p05.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P05" / "cut_roasted_beef" / "b0.5"
    rows = []
    with gpu_lock():
        adapter, scene = load_reference(load_cameras=True)
        k1_meta = build_k1(adapter, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], ROOT / "research_cache" / "cut_roasted_beef")
        k4_meta = build_k4(adapter, [0, 74, 149, 224, 299], ROOT / "research_cache" / "cut_roasted_beef")
        k1 = torch.load(k1_meta["path"], map_location="cpu")
        k4 = torch.load(k4_meta["path"], map_location="cpu")
        build_verified_ground_truth_cache(adapter, scene, splits["development"], samples["T24"], ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv")
        checks = p05.correctness_checks()
        if not checks["passed"]:
            raise RuntimeError(checks)
        combined_meta = {"key": f"{k1_meta['key']}+{k4_meta['key']}", "path": k4_meta["path"]}
        matrix = [("R0", seed) for seed in config["seeds"]] + [("R1", 0)]
        for rule, seed in matrix:
            run_root = base / f"seed{seed}"
            run_root.mkdir(parents=True, exist_ok=True)
            p01_runner.RUN_ROOT = run_root
            resolved = {**config, "seed": seed, "k1_key": k1_meta["key"], "k4_key": k4_meta["key"]}
            print(f"P05 {rule} seed={seed}: selecting", flush=True)
            selection = p05.select(adapter, k1, k4, rule, config["budget_fraction"], seed)
            report = f"# P05 {rule}\n\n状态：`COMPLETED`。零微调、50% 点预算。R0 为轨迹设施选址 stochastic greedy；R1 为相同稀疏邻域的 singleton 排序对照。\n"
            row = p01_runner._run_one(
                rule, resolved, adapter, scene, {}, combined_meta, splits, samples, checks,
                selection_result=selection, method_report_text=report,
            )
            row["seed"] = seed
            row["selection_diagnostics"] = selection.diagnostics
            rows.append(row)
            print(f"P05 {rule} seed={seed}: PSNR={row['mean_psnr']:.6f}", flush=True)
    randomized = [row for row in rows if row["rule"] == "R0"]
    result = {
        "status": "COMPLETED",
        "runs": rows,
        "r0_mean_psnr": statistics.fmean(row["mean_psnr"] for row in randomized),
        "r0_std_psnr_population": statistics.pstdev(row["mean_psnr"] for row in randomized),
        "r1_psnr": next(row["mean_psnr"] for row in rows if row["rule"] == "R1"),
    }
    base.mkdir(parents=True, exist_ok=True)
    write_json(base / "aggregate.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
