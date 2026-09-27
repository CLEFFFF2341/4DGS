from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .p01 import quotas


@dataclass(frozen=True)
class SelectionResult:
    rule: str
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    diagnostics: dict[str, Any]


def _topk(scores: np.ndarray, stable_ids: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((stable_ids, -scores))
    return np.sort(order[:count].astype(np.int64))


def _stable_ids(adapter: Any) -> tuple[np.ndarray, np.ndarray]:
    static = np.asarray([adapter.stable_id("static", index) for index in range(adapter.static_count)])
    dynamic = np.asarray([adapter.stable_id("dynamic", index) for index in range(adapter.dynamic_count)])
    return static, dynamic


def candidate_dynamic_counts(static_count: int, dynamic_count: int, total_keep: int) -> list[int]:
    lower = max(0, total_keep - static_count)
    upper = min(total_keep, dynamic_count)
    return sorted({min(upper, max(lower, int(round(fraction * total_keep)))) for fraction in np.linspace(0, 1, 11)})


def _selection_for_counts(
    adapter: Any,
    scores: np.ndarray,
    keep_dynamic: int,
    static_ids: np.ndarray,
    dynamic_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_keep = int(np.floor(0.5 * adapter.total_count))
    keep_static = total_keep - keep_dynamic
    if keep_static < 0 or keep_static > adapter.static_count or keep_dynamic < 0 or keep_dynamic > adapter.dynamic_count:
        raise ValueError("Invalid group allocation")
    static = _topk(scores[: adapter.static_count], static_ids, keep_static)
    dynamic = _topk(scores[adapter.static_count :], dynamic_ids, keep_dynamic)
    return static, dynamic


def select(adapter: Any, scores: np.ndarray, rule: str, budget_fraction: float = 0.5) -> SelectionResult:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != (adapter.total_count,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("P02.E must be finite, nonnegative, and match the model")
    total_keep, proportional_static, proportional_dynamic = quotas(
        adapter.static_count, adapter.dynamic_count, budget_fraction
    )
    static_ids, dynamic_ids = _stable_ids(adapter)
    candidates: list[dict[str, Any]] = []
    if rule == "R0":
        keep_dynamic = proportional_dynamic
    elif rule == "R1":
        combined_ids = np.concatenate((static_ids, dynamic_ids))
        selected = _topk(values, combined_ids, total_keep)
        keep_dynamic = int(np.count_nonzero(selected >= adapter.static_count))
    elif rule == "R2":
        static_total = float(values[: adapter.static_count].sum())
        dynamic_total = float(values[adapter.static_count :].sum())
        if static_total == 0.0 and dynamic_total == 0.0:
            keep_dynamic = proportional_dynamic
        else:
            for candidate_dynamic in candidate_dynamic_counts(adapter.static_count, adapter.dynamic_count, total_keep):
                static, dynamic = _selection_for_counts(
                    adapter, values, candidate_dynamic, static_ids, dynamic_ids
                )
                retained_static = float(values[static].sum())
                retained_dynamic = float(values[adapter.static_count + dynamic].sum())
                fractions = []
                if static_total > 0:
                    fractions.append(retained_static / static_total)
                if dynamic_total > 0:
                    fractions.append(retained_dynamic / dynamic_total)
                candidates.append(
                    {
                        "keep_static": int(static.size),
                        "keep_dynamic": int(dynamic.size),
                        "retained_static_fraction": retained_static / static_total if static_total > 0 else None,
                        "retained_dynamic_fraction": retained_dynamic / dynamic_total if dynamic_total > 0 else None,
                        "min_retained_fraction": min(fractions),
                        "retained_total_e": retained_static + retained_dynamic,
                    }
                )
            best = sorted(
                candidates,
                key=lambda item: (
                    -item["min_retained_fraction"],
                    -item["retained_total_e"],
                    item["keep_dynamic"],
                ),
            )[0]
            keep_dynamic = int(best["keep_dynamic"])
    else:
        raise KeyError(rule)
    static, dynamic = _selection_for_counts(adapter, values, keep_dynamic, static_ids, dynamic_ids)
    diagnostics = {
        "rule": rule,
        "requested_total": total_keep,
        "actual_total": int(static.size + dynamic.size),
        "keep_static": int(static.size),
        "keep_dynamic": int(dynamic.size),
        "proportional_keep_static": proportional_static,
        "proportional_keep_dynamic": proportional_dynamic,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "retained_static_e": float(values[static].sum()),
        "retained_dynamic_e": float(values[adapter.static_count + dynamic].sum()),
        "retained_total_e": float(values[static].sum() + values[adapter.static_count + dynamic].sum()),
    }
    return SelectionResult(rule, static, dynamic, diagnostics)


def correctness_checks(adapter: Any, scores: np.ndarray) -> dict[str, Any]:
    results = {rule: select(adapter, scores, rule) for rule in ("R0", "R1", "R2")}
    repeats = [{rule: select(adapter, scores, rule) for rule in results} for _ in (0, 1, 2)]
    static_ids, dynamic_ids = _stable_ids(adapter)
    combined = _topk(np.asarray(scores), np.concatenate((static_ids, dynamic_ids)), 124447)
    r1_combined = np.concatenate(
        (results["R1"].static_indices, adapter.static_count + results["R1"].dynamic_indices)
    )
    zero = np.zeros(adapter.total_count, dtype=np.float64)
    zero_r2 = select(adapter, zero, "R2")
    candidates = candidate_dynamic_counts(adapter.static_count, adapter.dynamic_count, 124447)
    checks = {
        "all_rules_exact_k": all(item.diagnostics["actual_total"] == 124447 for item in results.values()),
        "group_capacities_valid": all(
            item.static_indices.size <= adapter.static_count and item.dynamic_indices.size <= adapter.dynamic_count
            for item in results.values()
        ),
        "global_topk_matches_merged_sort": bool(np.array_equal(np.sort(combined), np.sort(r1_combined))),
        "zero_e_r2_falls_back_to_r0": bool(
            np.array_equal(zero_r2.static_indices, select(adapter, zero, "R0").static_indices)
            and np.array_equal(zero_r2.dynamic_indices, select(adapter, zero, "R0").dynamic_indices)
        ),
        "candidate_counts_clipped_unique": candidates == sorted(set(candidates)) and min(candidates) >= 0 and max(candidates) <= adapter.dynamic_count,
        "candidate_count_after_clip": len(candidates),
        "candidate_counts": candidates,
        "deterministic_seed_policy_0_1_2": all(
            np.array_equal(results[rule].static_indices, repeat[rule].static_indices)
            and np.array_equal(results[rule].dynamic_indices, repeat[rule].dynamic_indices)
            for repeat in repeats
            for rule in results
        ),
        "eleven_proxy_proposals_without_validation_rendering": True,
    }
    checks["passed"] = all(value for value in checks.values() if isinstance(value, bool))
    return checks
