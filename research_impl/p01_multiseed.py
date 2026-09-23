from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch

from .adapter import load_reference
from .cache import build_k1
from .config import ROOT
from .evaluate import build_verified_ground_truth_cache
from .methods.p01 import correctness_checks, select
from .runner import gpu_lock
from . import p01_runner


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    config = json.loads((ROOT / "configs" / "research" / "p01.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    base = ROOT / "runs" / "research" / "P01" / "cut_roasted_beef" / "b0.5"
    all_rows = []
    with gpu_lock():
        adapter, scene = load_reference(load_cameras=True)
        cache_meta = build_k1(
            adapter, scene, splits["c4"], samples["T24"], samples["statistics_resolution"],
            ROOT / "research_cache" / "cut_roasted_beef",
        )
        cache = torch.load(cache_meta["path"], map_location="cpu")
        build_verified_ground_truth_cache(
            adapter, scene, splits["development"], samples["T24"],
            ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv",
        )
        checks = correctness_checks(adapter, cache)
        if not checks["passed"]:
            raise RuntimeError(f"P01 correctness checks failed: {checks}")

        deterministic = {}
        for rule in config["deterministic_rules"]:
            selections = [select(adapter, cache, rule, config["budget_fraction"], seed) for seed in config["seeds"]]
            identical = all(
                np.array_equal(selections[0].static_indices, item.static_indices)
                and np.array_equal(selections[0].dynamic_indices, item.dynamic_indices)
                for item in selections[1:]
            )
            deterministic[rule] = {"seed_invariant_ids": bool(identical), "evaluated_seed": 0}
            if not identical:
                raise RuntimeError(f"Deterministic rule {rule} changed with seed")

        for seed in (1, 2):
            run_root = base / f"seed{seed}"
            run_root.mkdir(parents=True, exist_ok=True)
            p01_runner.RUN_ROOT = run_root
            seed_config = {**config, "seed": seed}
            seed_rows = []
            for rule in config["randomized_rules"]:
                print(f"P01 {rule} seed={seed}: start", flush=True)
                row = p01_runner._run_one(
                    rule, seed_config, adapter, scene, cache, cache_meta, splits, samples, checks
                )
                seed_rows.append(row)
                all_rows.append({**row, "seed": seed})
                print(f"P01 {rule} seed={seed}: PSNR={row['mean_psnr']:.6f}", flush=True)
            _json(run_root / "aggregate.json", {"status": "COMPLETED", "seed": seed, "runs": seed_rows})

    # Include the already completed seed-0 randomized rows.
    seed0 = json.loads((base / "seed0" / "aggregate.json").read_text(encoding="utf-8"))
    for row in seed0["runs"]:
        if row["rule"] in config["randomized_rules"]:
            all_rows.append({**row, "seed": 0})
    grouped = {}
    for rule in config["randomized_rules"]:
        rows = [row for row in all_rows if row["rule"] == rule]
        grouped[rule] = {
            "seeds": config["seeds"],
            "mean_psnr": statistics.fmean(row["mean_psnr"] for row in rows),
            "std_psnr_population": statistics.pstdev(row["mean_psnr"] for row in rows),
            "mean_lpips_alex": statistics.fmean(row["mean_lpips_alex"] for row in rows),
            "std_lpips_population": statistics.pstdev(row["mean_lpips_alex"] for row in rows),
            "mean_worst_10pct_psnr_drop": statistics.fmean(row["worst_10pct_mean_psnr_drop"] for row in rows),
            "runs": [{"seed": row["seed"], "config_hash": row["config_hash"], "psnr": row["mean_psnr"]} for row in rows],
        }
    result = {
        "status": "COMPLETED",
        "randomized_rules": grouped,
        "deterministic_seed_checks": deterministic,
        "note": "Only genuinely randomized R0/R5 are evaluated for seeds 0/1/2; R1-R4 are seed invariant.",
    }
    _json(base / "multiseed.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
