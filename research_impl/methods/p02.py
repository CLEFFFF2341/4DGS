from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .p01 import quotas


@dataclass(frozen=True)
class SelectionResult:
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    scores: np.ndarray
    diagnostics: dict


def _topk(scores: np.ndarray, stable_rows: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((stable_rows, -scores))
    return np.sort(order[:count].astype(np.int64))


def select(adapter, cache: dict, budget_fraction: float) -> SelectionResult:
    e_it = cache["e_it"].double().numpy()
    if e_it.shape[1] != adapter.total_count:
        raise ValueError(f"K2 point count {e_it.shape[1]} != model count {adapter.total_count}")
    if not np.isfinite(e_it).all() or np.any(e_it < 0):
        raise ValueError("K2 contains negative, NaN, or infinite deletion MSE")
    scores = e_it.mean(axis=0)
    total, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, budget_fraction)
    static = _topk(scores[: adapter.static_count], adapter.static_rows, keep_static)
    dynamic = _topk(scores[adapter.static_count :], adapter.dynamic_rows, keep_dynamic)
    return SelectionResult(
        static_indices=static,
        dynamic_indices=dynamic,
        scores=scores,
        diagnostics={
            "requested_total": total,
            "actual_total": int(static.size + dynamic.size),
            "static_quota": keep_static,
            "dynamic_quota": keep_dynamic,
            "nonzero_scores": int(np.count_nonzero(scores)),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
        },
    )


def correctness_checks(adapter, cache: dict) -> dict:
    selection = select(adapter, cache, 0.5)
    repeats = [select(adapter, cache, 0.5) for _ in (0, 1, 2)]
    e_it = cache["e_it"].double().numpy()
    checks = {
        "e_it_nonnegative_finite": bool(np.isfinite(e_it).all() and np.all(e_it >= 0)),
        "mean_matches_cache": bool(np.array_equal(selection.scores, e_it.mean(axis=0))),
        "quota_total": selection.diagnostics["actual_total"] == 124447,
        "static_quota": int(selection.static_indices.size) == 92516,
        "dynamic_quota": int(selection.dynamic_indices.size) == 31931,
        "unique_static": len(np.unique(selection.static_indices)) == len(selection.static_indices),
        "unique_dynamic": len(np.unique(selection.dynamic_indices)) == len(selection.dynamic_indices),
        "seed_invariant_0_1_2": bool(
            all(np.array_equal(selection.static_indices, item.static_indices) for item in repeats)
            and all(np.array_equal(selection.dynamic_indices, item.dynamic_indices) for item in repeats)
        ),
        "replay_max_abs": float(cache["replay_max_abs"]),
    }
    checks["passed"] = bool(all(value for key, value in checks.items() if isinstance(value, bool)) and checks["replay_max_abs"] <= 1e-4)
    return checks
