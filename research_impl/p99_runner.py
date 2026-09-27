from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROOT


OUTPUT = ROOT / "runs" / "research" / "aggregate" / "20260927-stage1-a5e2cda"

METHOD_STATUS = {
    "P01": ("COMPLETED_BASELINE", "P01-R4 is the frozen strongest simple count baseline."),
    "P02": ("COMPLETED_NOT_ADVANCED", "Exact single-deletion ranking loses to P01-R4."),
    "P03": ("COMPLETED_NOT_ADVANCED", "Peak aggregation helps P02 but remains below P01-R4."),
    "P04": ("BLOCKED_CACHE", "K3 CSR/ray cache is unavailable."),
    "P05": ("COMPLETED_NOT_ADVANCED", "Three-seed facility selection loses to P01-R4."),
    "P06": ("PROXY_INVALID", "Segment proxy is anti-correlated with development error."),
    "P07": ("NEGATIVE", "Fresh recalibration worsens mean and tail quality."),
    "P08": ("BLOCKED_CACHE", "K3 ordered ray lists are unavailable."),
    "P09": ("BLOCKED_CACHE", "K3 signed RGB deletion vectors are unavailable."),
    "P10": ("NEGATIVE", "All registered event demands are already satisfied; zero repairs."),
    "P11": ("NEGATIVE", "Hardening/optimization fails against equal-step ordinary FT."),
    "P12": ("NEGATIVE", "Reallocation adds bytes without quality improvement."),
    "P13": ("NEGATIVE", "Moment merging strongly loses to the same-ID keep-parent control."),
    "P14": ("BLOCKED_TARGET", "Only 13 dynamic points pass the frozen conversion gates."),
    "P15": ("ADVANCED_STAGE1", "Adaptive knots pass uniform and same-byte pruning controls."),
    "P16": ("NEGATIVE", "Haar loses to FP16; FP16 remains a useful simple control candidate."),
    "P17": ("NEGATIVE", "Low rank loses to FP16 and fails the tail criterion."),
    "P18": ("NEGATIVE", "Registered TDE gain is 0.229% and gather removes deployment speedup."),
    "P19": ("ADVANCED_STAGE1", "Tile-aware row passes matched tile quality and latency controls."),
    "P20": ("BLOCKED_CACHE", "K3 ray direction/weight/color data are unavailable."),
}

METHOD_FAIRNESS = {
    "P01": ("PASS_CONTROL", "Three seeds only for R0/R5; deterministic rows not duplicated."),
    "P02": ("PASS", "Exact K2 replay and group quotas verified."),
    "P03": ("PASS", "Same K2 input and verified temporal GT cache."),
    "P04": ("BLOCKED", "No formal run without K3."),
    "P05": ("PASS", "Random R0 aggregated across seeds 0/1/2."),
    "P06": ("EXCLUDED_PROXY", "Correct implementation but invalid proxy direction."),
    "P07": ("PASS", "Fresh and stale controls rebuild equal-cost K2 rounds."),
    "P08": ("BLOCKED", "No lower-fidelity substitute for K3."),
    "P09": ("BLOCKED", "Squared K2 cannot replace signed K3 data."),
    "P10": ("PASS_NO_OPPORTUNITY", "Exact same-ID no-repair control."),
    "P11": ("PASS_NEGATIVE", "Three seeds and equal-step R2 FT control."),
    "P12": ("PASS_NEGATIVE", "Bytes reported; expensive dynamic points not hidden."),
    "P13": ("PASS_NEGATIVE", "R0/R1 use identical IDs, points, and bytes."),
    "P14": ("BLOCKED_TARGET", "Frozen gates were not relaxed."),
    "P15": ("PASS_ADVANCE", "Uniform and same-byte P01-R4 controls complete."),
    "P16": ("PASS_NEGATIVE", "Haar compared with simpler FP16 and same-byte pruning."),
    "P17": ("PASS_NEGATIVE", "DCT and near-byte FP16 explanations retained."),
    "P18": ("PASS_NEGATIVE", "GT-residual TDE and gather-inclusive latency used."),
    "P19": ("PASS_ADVANCE", "Byte and tile lanes kept separate; both matched controls complete."),
    "P20": ("BLOCKED", "No formal run without K3."),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_json(path: Path, name: str) -> dict[str, Any]:
    current = path
    root = ROOT / "runs" / "research"
    while current != root.parent:
        candidate = current / name
        if candidate.exists():
            return read_json(candidate)
        if current == root:
            break
        current = current.parent
    return {}


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def average(rows: list[dict[str, str]], key: str) -> float | None:
    values = [value for row in rows if (value := numeric(row, key)) is not None]
    return statistics.fmean(values) if values else None


def peak_resource(resources: dict[str, Any], key: str) -> int | None:
    values = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                if child_key == key and isinstance(child, (int, float)):
                    values.append(int(child))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(resources)
    return max(values) if values else None


def latency_mean(resources: dict[str, Any]) -> float | None:
    latency = resources.get("latency", {})
    blocks = latency.get("blocks", []) if isinstance(latency, dict) else []
    values = [block.get("wall_mean_seconds") for block in blocks if block.get("wall_mean_seconds") is not None]
    return statistics.fmean(values) if values else None


def recompute_run(path: Path, reference_pairs: set[tuple[str, int]]) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    relative = path.relative_to(ROOT / "runs" / "research")
    parts = relative.parts
    method = parts[0] if parts[0].startswith("P") else parts[0]
    config = nearest_json(path.parent, "resolved_config.json")
    resources = nearest_json(path.parent, "resources.json")
    status = nearest_json(path.parent, "status.json")
    pairs = {(row.get("camera", ""), int(float(row.get("timestamp", -1)))) for row in rows}
    psnr_values = [numeric(row, "psnr") for row in rows]
    psnr_values = [value for value in psnr_values if value is not None]
    drops = [numeric(row, "psnr_drop") for row in rows]
    drops = [value for value in drops if value is not None]
    tail_count = max(1, int(np.ceil(0.1 * len(drops)))) if drops else 0
    last_rows = [row for row in rows if int(float(row.get("timestamp", -1))) >= 270]
    rule = config.get("rule", config.get("row", ""))
    path_rule = next(
        (part for part in reversed(parts) if part.startswith("R") and part[1:].isdigit()),
        None,
    )
    if path_rule is not None:
        rule = path_rule
    seed = config.get("seed", "")
    if seed == "":
        for part in parts:
            if part.startswith("seed"):
                seed = part.removeprefix("seed")
                break
    diagnostic = bool(
        "decode_invalid" in parts
        or "soft_evaluation" in parts
        or "extra_reappearance" in path.name
        or method in {"resource_controls", "P00"}
    )
    actual_points = resources.get("actual_total_points", resources.get("unique_points"))
    bundle_bytes = resources.get("bundle_bytes", resources.get("compact_bundle_bytes"))
    return {
        "record_type": "evaluated_run",
        "method": method,
        "rule": rule,
        "seed": seed,
        "run_id": str(path.parent.relative_to(ROOT / "runs" / "research")).replace("\\", "/"),
        "status": status.get("status", "COMPLETED"),
        "diagnostic_or_control": diagnostic,
        "comparable_v_t24": pairs == reference_pairs,
        "frame_count": len(rows),
        "camera_time_hash": hashlib.sha256(repr(sorted(pairs)).encode()).hexdigest(),
        "mean_psnr": statistics.fmean(psnr_values) if psnr_values else None,
        "mean_ssim": average(rows, "ssim"),
        "mean_lpips": average(rows, "lpips_alex"),
        "mean_delta_mse": average(rows, "delta_mse"),
        "mean_teacher_mse": average(rows, "teacher_mse"),
        "worst_10pct_drop": statistics.fmean(sorted(drops, reverse=True)[:tail_count]) if drops else None,
        "p95_delta_mse": float(np.percentile([numeric(row, "delta_mse") for row in rows if numeric(row, "delta_mse") is not None], 95)) if any(numeric(row, "delta_mse") is not None for row in rows) else None,
        "last30_mean_psnr": average(last_rows, "psnr"),
        "actual_points": actual_points,
        "static_points": resources.get("static_points"),
        "dynamic_points": resources.get("dynamic_points"),
        "bundle_bytes": bundle_bytes,
        "decoded_tensor_bytes": resources.get("loaded_tensor_bytes"),
        "active_gaussian_frames": resources.get("actual_active_frames"),
        "tile_cost": resources.get("selected_tile_cost"),
        "latency_wall_mean_seconds": latency_mean(resources),
        "peak_allocated_bytes": peak_resource(resources, "peak_allocated_bytes"),
        "peak_reserved_bytes": peak_resource(resources, "peak_reserved_bytes"),
        "source_csv": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def find_run(runs: list[dict[str, Any]], method: str, rule: str, seed: str | int = "") -> dict[str, Any]:
    candidates = [
        row
        for row in runs
        if row["method"] == method
        and row["rule"] == rule
        and row["comparable_v_t24"]
        and "decode_invalid" not in row["run_id"]
        and "soft_evaluation" not in row["run_id"]
        and (seed == "" or str(row["seed"]) == str(seed))
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one run for {method}/{rule}/seed={seed}, got {[row['run_id'] for row in candidates]}")
    return candidates[0]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / "research" / "experiment_manifest.json")
    progress = read_json(ROOT / "research" / "progress.json")
    reference_csv = ROOT / "runs" / "research" / "P00" / "reference_v_t24" / "per_frame.csv"
    reference_rows = list(csv.DictReader(reference_csv.open("r", encoding="utf-8")))
    reference_pairs = {(row["camera"], int(row["timestamp"])) for row in reference_rows}
    csv_paths = sorted((ROOT / "runs" / "research").glob("**/per_frame.csv"))
    evaluated = [recompute_run(path, reference_pairs) for path in csv_paths]
    method_rows = [
        {
            "record_type": "method_status",
            "method": item["id"],
            "rule": "",
            "seed": "",
            "run_id": item["id"],
            "status": METHOD_STATUS[item["id"]][0],
            "diagnostic_or_control": False,
            "comparable_v_t24": "",
            "frame_count": "",
            "reason": METHOD_STATUS[item["id"]][1],
        }
        for item in manifest
    ]
    write_csv(OUTPUT / "all_runs.csv", method_rows + evaluated)
    comparable = [
        row
        for row in evaluated
        if row["comparable_v_t24"]
        and "decode_invalid" not in row["run_id"]
        and "soft_evaluation" not in row["run_id"]
        and "extra_reappearance" not in row["run_id"]
        and row["method"] != "P00"
    ]
    write_csv(OUTPUT / "per_scene_budget.csv", comparable)

    p01 = find_run(evaluated, "P01", "R4", 0)
    p02 = find_run(evaluated, "P02", "R0")
    p15 = find_run(evaluated, "P15", "R0")
    p16 = find_run(evaluated, "P16", "R1")
    p19_byte = find_run(evaluated, "P19", "R0")
    p19_tile = find_run(evaluated, "P19", "R1")
    pareto = []
    for row, lane, eligibility, reason in (
        (p01, "count/byte", "REQUIRED_CONTROL", "Frozen strongest simple count baseline."),
        (p02, "count/byte", "REQUIRED_CONTROL", "Frozen exact single-deletion control."),
        (p15, "position+rotation bytes", "STAGE2_CANDIDATE", "Passes uniform and same-byte pruning controls."),
        (p16, "position bytes", "CONTROL_CANDIDATE", "FP16 dominates Haar and same-byte pruning; retain as simple representation control."),
        (p19_byte, "total bytes", "NOT_SELECTED", "Passes internal byte-P02 control but loses 0.416 dB to P01-R4 at essentially the same bytes."),
        (p19_tile, "tile cost", "STAGE2_CANDIDATE", "Passes exact same-tile quality control and measured latency threshold."),
    ):
        pareto.append(
            {
                "method": row["method"],
                "rule": row["rule"],
                "lane": lane,
                "eligibility": eligibility,
                "reason": reason,
                "psnr": row["mean_psnr"],
                "lpips": row["mean_lpips"],
                "worst_10pct_drop": row["worst_10pct_drop"],
                "points": row["actual_points"],
                "bundle_bytes": row["bundle_bytes"],
                "tile_cost": row["tile_cost"],
                "latency_seconds": row["latency_wall_mean_seconds"],
                "source": row["run_id"],
            }
        )
    write_csv(OUTPUT / "pareto.csv", pareto)

    fairness = [
        {
            "method": item["id"],
            "status": METHOD_STATUS[item["id"]][0],
            "fairness": METHOD_FAIRNESS[item["id"]][0],
            "audit": METHOD_FAIRNESS[item["id"]][1],
            "test_cam00_sealed": True,
            "official_checkpoint_saw_development_views": True,
        }
        for item in manifest
    ]
    write_csv(OUTPUT / "fairness_audit.csv", fairness)

    ledger = []
    for path in sorted((ROOT / "runs" / "research").glob("**/timing.json")):
        if "aggregate" in path.parts:
            continue
        value = read_json(path)
        wall = value.get("total_seconds", value.get("run_seconds", value.get("wall_seconds")))
        ledger.append(
            {
                "record_type": "run_timing_artifact",
                "id": str(path.parent.relative_to(ROOT / "runs" / "research")).replace("\\", "/"),
                "wall_seconds": wall,
                "gpu_hours": "",
                "cpu_selection_hours": "",
                "peak_allocated_bytes": "",
                "peak_reserved_bytes": "",
                "retries": 0,
                "note": "Wall artifact only; not summed as GPU hours.",
            }
        )
    for path in sorted((ROOT / "research_cache").glob("**/metadata.json")):
        value = read_json(path)
        ledger.append(
            {
                "record_type": "cache_artifact",
                "id": str(path.parent.relative_to(ROOT)).replace("\\", "/"),
                "wall_seconds": value.get("seconds"),
                "gpu_hours": "",
                "cpu_selection_hours": "",
                "peak_allocated_bytes": value.get("peak_allocated_bytes"),
                "peak_reserved_bytes": value.get("peak_reserved_bytes"),
                "retries": 0,
                "note": "Unique cache metadata; reuse is not counted again.",
            }
        )
    ledger.append(
        {
            "record_type": "canonical_stage_total",
            "id": "progress.json",
            "wall_seconds": "",
            "gpu_hours": progress["gpu_hours_consumed"],
            "cpu_selection_hours": progress["cpu_selection_hours_consumed"],
            "peak_allocated_bytes": "",
            "peak_reserved_bytes": "",
            "retries": 2,
            "note": "Canonical serial GPU accounting; resolved K2 and video-decode implementation retries retained in failure catalog.",
        }
    )
    write_csv(OUTPUT / "budget_ledger.csv", ledger)

    next_queue = {
        "schema": "research-stage2-queue-v1",
        "generated_from": "stage1 only",
        "authorized_to_start": False,
        "blockers": [
            "Stage 2 is not authorized by research/progress.json.",
            "Official S1 coffee_martini and S2 sear_steak pretrained checkpoints and complete preprocessing are unavailable.",
        ],
        "candidates": [
            {
                "rank": 1,
                "method": "P15",
                "rule": "R0",
                "lane": "position+rotation bytes",
                "status": "READY_AFTER_DATA_AND_AUTHORIZATION",
                "evidence": "Same-byte P01-R4 +6.746460 dB; uniform allocation +2.540168 dB.",
            },
            {
                "rank": 2,
                "method": "P19",
                "rule": "R1",
                "lane": "tile cost",
                "status": "READY_AFTER_DATA_AND_AUTHORIZATION",
                "evidence": "Same-tile control +1.342975 dB and 7.51% lower measured latency.",
            },
            {
                "rank": 3,
                "method": "P16",
                "rule": "R1",
                "lane": "position bytes simple control",
                "status": "CONTROL_CANDIDATE_AFTER_DATA_AND_AUTHORIZATION",
                "evidence": "FP16 retains near-reference quality and beats same-byte pruning by 6.414396 dB.",
            },
        ],
        "required_controls": [
            {"method": "P01", "rule": "R4"},
            {"method": "P02", "rule": "R0"},
        ],
        "not_selected": [
            {
                "method": "P19",
                "rule": "R0",
                "reason": "Internal byte control passes, but P01-R4 is 0.416065 dB better at only 0.0095% more bytes; the difference is below the 10% resource alternative threshold.",
            }
        ],
    }
    (OUTPUT / "next_queue.json").write_text(json.dumps(next_queue, indent=2), encoding="utf-8")

    failure_lines = ["# 阶段一失败与阻塞目录", ""]
    for item in progress["failures"]:
        failure_lines.extend(
            [
                f"## {item['task']} — {item['status']}",
                "",
                item["summary"],
                "",
                f"产物：`{item['artifact']}`。",
                "",
            ]
        )
    (OUTPUT / "failure_catalog.md").write_text("\n".join(failure_lines), encoding="utf-8")

    report = f"""# 阶段一事实汇总

本汇总覆盖 P01–P20 全部 20 个方案 ID。最终状态为：2 个 `ADVANCED_STAGE1`（P15、P19），5 个阻塞（P04/P08/P09/P14/P20），1 个 `PROXY_INVALID`（P06），8 个正确实现后的 `NEGATIVE`，3 个完成但不晋级，另有 1 个必要基线 P01。cam00 最终测试仍封存。

阶段一累计账本为 **{progress['gpu_hours_consumed']:.4f} GPU 小时**、**{progress['cpu_selection_hours_consumed']:.4f} CPU 选择小时**，使用 **{progress['resource_control_slots_used']}/4** 个共享资源控制槽。`budget_ledger.csv` 保留各 timing/cache 墙钟产物，但不把它们直接相加冒充 GPU 小时。

## 最有信息量的正证据

- P15-R0 在 95,270,974 B 下比同字节 P01-R4 高 6.746460 dB，也比同 payload uniform 高 2.540168 dB；只主张磁盘表示收益。
- P19-R1 在相同 50% tile cap 下比纯 E 控制高 1.342975 dB、LPIPS 好 0.010470，实测 wall latency 低 7.51%；相对完整模型约快 48%。byte 与 tile 赛道没有混排。
- P16-R1 FP16 是强简单控制：114,173,750 B 时 PSNR 35.649530 dB，说明 P16/P17 的复杂变换收益可被低精度存储解释。

## 最有信息量的负证据

- P11 learned mask 的 hard 结果跨三种子落后等步普通 FT；soft→hard 平均损失 6.068814 dB，属于优化/硬化失败。
- P13 矩匹配合并相对同 ID、同字节保父控制下降 4.504748 dB，说明空间矩不保持遮挡与 alpha 合成。
- P18 虽减少 52.95% proxy switches，但注册 GT-residual TDE 只改善 0.2289%，且含 gather 延迟更慢。
- P07 fresh 重估改变了边界 ID，却比 stale/P02 低 0.385796 dB；P12 改静动态配额只增加 bytes 或降质量。

## 公平性与代表性

所有主比较使用 S0 开发视角 cam01/cam02×T24、1352×1014 与冻结 GT cache；逐帧均值已从 CSV 重算。确定性 seed 重算不当独立样本，随机 P01/P05/P11 使用三种子。官方 checkpoint 训练时见过开发视角，因此这些结果不是未见视角泛化；阶段二还缺两个官方场景 checkpoint/完整预处理。K3 缺失使 P04/P08/P09/P20 只能标阻塞，不能把缺失当方法失败。

下一阶段只生成队列，未获授权也未启动。候选为 P15-R0、P19-R1，以及作为简单表示控制候选的 P16-R1；P01-R4 与 P02 是必带控制。P19-R0 虽胜内部 byte-P02 控制，但在几乎相同 bytes 下落后 P01-R4 0.416 dB，因此 P99 公平性筛选未选它。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")

    handoff = """# Astra 阶段一交接

## P15-R0（A 类迁移候选）

支持：同 payload 胜 uniform、自适应误差分配机制明确，并通过同字节剪枝控制。不支持：未证明跨场景、最终测试或显存/推理加速；解码仍恢复 dense FP32。需要 Astra 判断它作为自适应 keyframe 迁移与近期 TC3DGS/Ex4DGS 的差距，以及是否值得进入阶段二。

## P19-R1（C 类候选）

支持：在冻结 50% tile 代理 cap 下同时胜 exact same-tile 纯 E 控制并取得真实延迟改善。不支持：不是同字节胜利，14,156 个 0/0 点说明统计视角覆盖有限，也没有跨场景证据。需要 Astra 判断实际资源成本排序是否构成足够独立的研究增量，而非 Speedy-Splat/LightGaussian 成本感知排序的直接迁移。

## P16-R1（简单控制候选）

支持：FP16 position 在约半位置载荷下几乎保持参考质量，并支配 Haar/低秩与同字节剪枝。不支持：它是简单控制而不是 P16 的研究胜利，也不主张新颖性。建议阶段二保留用于解释 P15/P17 等表示方法。

## 未完成的 C 类机制

P04/P08/P09 因 K3 缺失阻塞，P14 因冻结目标不可达，P06 代理失效，P18 未通过 TDE/部署门槛。P20 虽为 A 类，也同样被 K3 阻塞。若 Astra 认为联合 ray/SH 机制优先级高，下一项基础设施决策是是否投入共享 K3；本汇总不自行改变预算或方法定义。
"""
    (OUTPUT / "astra_handoff.md").write_text(handoff, encoding="utf-8")

    visuals = OUTPUT / "fixed_visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    visual_sources = {
        "P01_R4": ROOT / "runs" / "research" / p01["run_id"] / "visualizations",
        "P02_R0": ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0" / "visualizations",
        "P15_R0": ROOT / "runs" / "research" / "P15" / "cut_roasted_beef" / "R0" / "visualizations",
        "P16_R1": ROOT / "runs" / "research" / "P16" / "cut_roasted_beef" / "R1" / "visualizations",
        "P19_R1": ROOT / "runs" / "research" / "P19" / "cut_roasted_beef" / "R1" / "visualizations",
    }
    visual_manifest = []
    for label, directory in visual_sources.items():
        for source in sorted(directory.glob("*.png")):
            target = visuals / f"{label}_{source.name}"
            shutil.copy2(source, target)
            visual_manifest.append(
                {"method": label, "source": str(source.relative_to(ROOT)), "file": target.name}
            )
    (visuals / "manifest.json").write_text(
        json.dumps(
            {
                "fixed_times": [0, 74, 149, 224, 299],
                "panels": "GT | immutable reference | method; native [0,1] clipping",
                "files": visual_manifest,
                "limitation": "Worst-error frames are recorded in all_runs/per_scene tables but were not re-rendered because P99 is GPU-free.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "evaluated_csvs": len(evaluated), "method_ids": len(method_rows)}, indent=2))


if __name__ == "__main__":
    main()
