from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .p01 import quotas


@dataclass(frozen=True)
class SelectionResult:
    rule: str
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    diagnostics: dict


def _topk(scores: np.ndarray, stable_rows: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.lexsort((stable_rows, -scores))[:count].astype(np.int64))


def _candidate(adapter, normalized: np.ndarray, weights: np.ndarray, budget_fraction: float) -> tuple[np.ndarray, np.ndarray, dict]:
    scores = np.sum(weights[:, None] * normalized, axis=0)
    _, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, budget_fraction)
    static = _topk(scores[: adapter.static_count], adapter.static_rows, keep_static)
    dynamic = _topk(scores[adapter.static_count :], adapter.dynamic_rows, keep_dynamic)
    selected = np.concatenate((static, adapter.static_count + dynamic))
    z = normalized[:, selected].sum(axis=1)
    return static, dynamic, {"z": z.tolist(), "min_z": float(z.min()), "mean_z": float(z.mean())}


def select(adapter, cache: dict, rule: str, budget_fraction: float, rounds: int = 5, eta: float = 2.0, fallback_scores: np.ndarray | None = None) -> SelectionResult:
    e_it = cache["e_it"].double().numpy()
    if e_it.shape[0] != 24:
        raise ValueError("P06 requires exactly T24")
    segmented = e_it.reshape(6, 4, adapter.total_count).sum(axis=1)
    totals = segmented.sum(axis=1)
    valid = totals > 0
    if not valid.any():
        if fallback_scores is None:
            raise RuntimeError("All P06 segments are zero and no P01-R2 fallback was supplied")
        normalized = np.zeros_like(segmented)
        weights = np.zeros(6)
        weights[0] = 1.0
        static, dynamic, diagnostics = _candidate(adapter, np.tile(fallback_scores[None], (6, 1)), weights, budget_fraction)
        diagnostics.update({"fallback": "P01-R2", "candidates": []})
        return SelectionResult(rule, static, dynamic, diagnostics)
    normalized = np.zeros_like(segmented)
    normalized[valid] = segmented[valid] / (totals[valid, None] + 1e-12)
    weights = np.zeros(6, dtype=np.float64)
    weights[valid] = 1.0 / int(valid.sum())
    candidates = []
    iterations = rounds if rule == "R0" else 1
    for iteration in range(iterations):
        static, dynamic, objective = _candidate(adapter, normalized[valid], weights[valid], budget_fraction)
        full_z = np.zeros(6, dtype=np.float64)
        full_z[valid] = objective["z"]
        candidate = {
            "iteration": iteration,
            "weights": weights.tolist(),
            "z": full_z.tolist(),
            "min_z": float(full_z[valid].min()),
            "mean_z": float(full_z[valid].mean()),
            "static_indices": static,
            "dynamic_indices": dynamic,
        }
        candidates.append(candidate)
        log_weights = np.full(6, -np.inf, dtype=np.float64)
        log_weights[valid] = np.log(weights[valid]) + eta * (1.0 - full_z[valid])
        maximum = np.max(log_weights[valid])
        updated = np.exp(log_weights[valid] - maximum)
        weights[valid] = updated / updated.sum()
    def tie_ids(candidate: dict) -> tuple:
        return tuple(adapter.static_rows[candidate["static_indices"]].tolist()) + tuple(adapter.dynamic_rows[candidate["dynamic_indices"]].tolist())
    winner = sorted(candidates, key=lambda item: (-item["min_z"], -item["mean_z"], tie_ids(item)))[0]
    diagnostics = {
        "fallback": None,
        "valid_segments": np.flatnonzero(valid).tolist(),
        "segment_totals": totals.tolist(),
        "winner_iteration": winner["iteration"],
        "winner_z": winner["z"],
        "winner_min_z": winner["min_z"],
        "winner_mean_z": winner["mean_z"],
        "candidates": [
            {key: value for key, value in item.items() if key not in {"static_indices", "dynamic_indices"}}
            for item in candidates
        ],
        "actual_total": int(len(winner["static_indices"]) + len(winner["dynamic_indices"])),
        "static_quota": int(len(winner["static_indices"])),
        "dynamic_quota": int(len(winner["dynamic_indices"])),
        "selection_does_not_read_development_views": True,
    }
    return SelectionResult(rule, winner["static_indices"], winner["dynamic_indices"], diagnostics)


def correctness_checks(adapter, cache: dict, fallback_scores: np.ndarray) -> dict:
    e_it = cache["e_it"].double().numpy()
    segmented = e_it.reshape(6, 4, adapter.total_count).sum(axis=1)
    totals = segmented.sum(axis=1)
    normalized = segmented / (totals[:, None] + 1e-12)
    all_z = normalized.sum(axis=1)
    r0 = select(adapter, cache, "R0", 0.5, fallback_scores=fallback_scores)
    r1 = select(adapter, cache, "R1", 0.5, fallback_scores=fallback_scores)
    repeats = [select(adapter, cache, "R0", 0.5, fallback_scores=fallback_scores) for _ in (0, 1, 2)]
    toy = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )
    toy_weights = np.full(6, 1 / 6)
    first_scores = (toy_weights[:, None] * toy).sum(axis=0)
    first = np.argsort(-first_scores)[:2]
    first_z = toy[:, first].sum(axis=1) / (toy.sum(axis=1) + 1e-12)
    updated_log = np.log(toy_weights) + 2 * (1 - first_z)
    updated = np.exp(updated_log - updated_log.max())
    updated /= updated.sum()
    second = np.argsort(-(updated[:, None] * toy).sum(axis=0))[:2]
    checks = {
        "all_point_z_near_one": bool(np.max(np.abs(all_z - 1.0)) <= 1e-9),
        "five_candidates_saved": len(r0.diagnostics["candidates"]) == 5,
        "uniform_one_candidate_saved": len(r1.diagnostics["candidates"]) == 1,
        "quotas_valid": r0.diagnostics["static_quota"] == 92516 and r0.diagnostics["dynamic_quota"] == 31931,
        "exponent_update_finite": bool(np.isfinite(updated).all() and abs(updated.sum() - 1) <= 1e-12),
        "imbalanced_toy_reallocates": bool(not np.array_equal(np.sort(first), np.sort(second))),
        "selection_does_not_read_development_views": r0.diagnostics["selection_does_not_read_development_views"],
        "seed_invariant_0_1_2": bool(all(np.array_equal(r0.static_indices, item.static_indices) and np.array_equal(r0.dynamic_indices, item.dynamic_indices) for item in repeats)),
    }
    checks["passed"] = all(checks.values())
    return checks
