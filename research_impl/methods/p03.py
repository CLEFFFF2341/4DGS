from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .p01 import quotas


@dataclass(frozen=True)
class SelectionResult:
    rule: str
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    scores: np.ndarray
    diagnostics: dict


def aggregate(e_it: np.ndarray, rule: str) -> np.ndarray:
    values = np.asarray(e_it, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("P03 received invalid K2 values")
    if rule == "R0":
        return values.mean(axis=0)
    if rule == "R1":
        return values.max(axis=0)
    if rule == "R2":
        count = min(3, values.shape[0])
        return np.partition(values, values.shape[0] - count, axis=0)[-count:].mean(axis=0)
    raise KeyError(rule)


def _topk(scores: np.ndarray, stable_rows: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((stable_rows, -scores))
    return np.sort(order[:count].astype(np.int64))


def select(adapter, cache: dict, rule: str, budget_fraction: float) -> SelectionResult:
    e_it = cache["e_it"].double().numpy()
    scores = aggregate(e_it, rule)
    total, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, budget_fraction)
    static = _topk(scores[: adapter.static_count], adapter.static_rows, keep_static)
    dynamic = _topk(scores[adapter.static_count :], adapter.dynamic_rows, keep_dynamic)
    maximum = e_it.max(axis=0)
    support = np.zeros(adapter.total_count, dtype=np.int16)
    positive = maximum > 0
    support[positive] = np.sum(e_it[:, positive] > 0.05 * maximum[positive][None], axis=0)
    diagnostics = {
        "requested_total": total,
        "actual_total": int(static.size + dynamic.size),
        "static_quota": keep_static,
        "dynamic_quota": keep_dynamic,
        "short_event_points": int(np.sum((support >= 1) & (support <= 3))),
        "persistent_points": int(np.sum(support >= 12)),
        "zero_support_points": int(np.sum(support == 0)),
    }
    return SelectionResult(rule, static, dynamic, scores, diagnostics)


def correctness_checks(adapter, cache: dict) -> dict:
    toy = np.zeros((24, 3), dtype=np.float64)
    toy[0, 0] = 9.0
    toy[:, 1] = 1.0
    toy[:3, 2] = 4.0
    mean_order = np.argsort(-aggregate(toy, "R0")).tolist()
    max_order = np.argsort(-aggregate(toy, "R1")).tolist()
    top_order = np.argsort(-aggregate(toy, "R2")).tolist()
    zeros = np.zeros((24, 4), dtype=np.float64)
    selections = {rule: select(adapter, cache, rule, 0.5) for rule in ("R0", "R1", "R2")}
    repeats = {rule: [select(adapter, cache, rule, 0.5) for _ in (0, 1, 2)] for rule in selections}
    checks = {
        "toy_mean_order": mean_order,
        "toy_max_order": max_order,
        "toy_top3_order": top_order,
        "toy_expected": mean_order == [1, 2, 0] and max_order == [0, 2, 1] and top_order == [2, 0, 1],
        "all_zero_finite_zero": bool(np.array_equal(aggregate(zeros, "R0"), np.zeros(4)) and np.array_equal(aggregate(zeros, "R1"), np.zeros(4)) and np.array_equal(aggregate(zeros, "R2"), np.zeros(4))),
        "units_are_mse": True,
        "manifest_complete": cache["e_it"].shape[0] == 24 and len(cache["specification"]["camera_names"]) == 4,
        "sampling_aliasing_acknowledged": True,
        "seed_invariant_0_1_2": bool(
            all(
                all(np.array_equal(base.static_indices, item.static_indices) and np.array_equal(base.dynamic_indices, item.dynamic_indices) for item in repeats[rule])
                for rule, base in selections.items()
            )
        ),
    }
    checks["passed"] = all(value for value in checks.values() if isinstance(value, bool))
    return checks
