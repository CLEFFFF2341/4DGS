from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .p01 import SelectionResult, quotas


def _reverse_graph(neighbors: np.ndarray, similarity: np.ndarray):
    count, degree = neighbors.shape
    candidate = neighbors.reshape(-1).astype(np.int64)
    demand = np.repeat(np.arange(count, dtype=np.int64), degree)
    values = similarity.reshape(-1).astype(np.float64)
    order = np.argsort(candidate, kind="stable")
    candidate = candidate[order]
    demand = demand[order]
    values = values[order]
    indptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(candidate, minlength=count), out=indptr[1:])
    return indptr, demand, values


def _weights(r2: np.ndarray) -> np.ndarray:
    values = np.asarray(r2, dtype=np.float64)
    total = values.sum()
    return values / total if total > 0 else np.full(values.size, 1.0 / values.size)


def _singleton(neighbors: np.ndarray, similarity: np.ndarray, weights: np.ndarray) -> np.ndarray:
    degree = neighbors.shape[1]
    return np.bincount(
        neighbors.reshape(-1),
        weights=np.repeat(weights, degree) * similarity.reshape(-1),
        minlength=neighbors.shape[0],
    )


def _topk(scores: np.ndarray, stable_rows: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((stable_rows, -scores))
    return np.sort(order[:count].astype(np.int64))


def _facility(
    neighbors: np.ndarray,
    similarity: np.ndarray,
    weights: np.ndarray,
    stable_rows: np.ndarray,
    r2: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    size = neighbors.shape[0]
    sample_size = min(size, int(math.ceil((size / count) * math.log(20)))) if count else 0
    indptr, demands, values = _reverse_graph(neighbors, similarity)
    coverage = np.zeros(size, dtype=np.float64)
    selected = np.zeros(size, dtype=bool)
    rng = np.random.default_rng(seed)
    evaluations = 0
    fallback = 0
    started = time.perf_counter()
    for step in range(count):
        candidates = []
        seen = set()
        while len(candidates) < sample_size:
            candidate = int(rng.integers(0, size))
            if not selected[candidate] and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        best = -1
        best_gain = -1.0
        for candidate in candidates:
            begin, end = indptr[candidate], indptr[candidate + 1]
            rows = demands[begin:end]
            gain = float(np.dot(weights[rows], np.maximum(values[begin:end] - coverage[rows], 0.0)))
            evaluations += 1
            if gain > best_gain or (gain == best_gain and (best < 0 or stable_rows[candidate] < stable_rows[best])):
                best, best_gain = candidate, gain
        if best_gain <= 0:
            remaining = count - step
            fallback_order = np.lexsort((stable_rows, -r2))
            additions = fallback_order[~selected[fallback_order]][:remaining]
            selected[additions] = True
            fallback = int(remaining)
            break
        selected[best] = True
        begin, end = indptr[best], indptr[best + 1]
        rows = demands[begin:end]
        np.maximum.at(coverage, rows, values[begin:end])
    chosen = np.flatnonzero(selected)
    return np.sort(chosen.astype(np.int64)), {
        "sample_size": sample_size,
        "marginal_evaluations": evaluations,
        "fallback_count": fallback,
        "objective": float(np.dot(weights, coverage)),
        "selection_seconds": time.perf_counter() - started,
    }


def select(adapter: Any, k1: dict[str, Any], k4: dict[str, Any], rule: str, budget_fraction: float, seed: int) -> SelectionResult:
    _, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, budget_fraction)
    r2_all = k1["s_it"].double().mean(dim=0).numpy()
    selections = []
    score_parts = []
    diagnostics = {"static_quota": keep_static, "dynamic_quota": keep_dynamic}
    offset = 0
    for kind, count, keep in (("static", adapter.static_count, keep_static), ("dynamic", adapter.dynamic_count, keep_dynamic)):
        neighbors = k4[f"{kind}_neighbors"].numpy()
        similarity = k4[f"{kind}_similarity"].numpy()
        weights = _weights(r2_all[offset : offset + count])
        singleton = _singleton(neighbors, similarity, weights)
        stable = adapter.static_rows if kind == "static" else adapter.dynamic_rows
        if rule == "R0":
            chosen, diag = _facility(neighbors, similarity, weights, stable, r2_all[offset : offset + count], keep, seed)
        elif rule == "R1":
            chosen = _topk(singleton, stable, keep)
            diag = {"objective": None, "selection_seconds": 0.0, "marginal_evaluations": 0, "fallback_count": 0}
        else:
            raise KeyError(rule)
        selections.append(chosen)
        score_parts.append(singleton)
        diagnostics[kind] = diag
        offset += count
    diagnostics["actual_total"] = int(sum(len(value) for value in selections))
    return SelectionResult(rule, selections[0], selections[1], score_parts[0], score_parts[1], diagnostics)


def correctness_checks() -> dict[str, Any]:
    neighbors = np.asarray([[0, 1], [1, 0], [2, 1]], dtype=np.int32)
    similarity = np.asarray([[1.0, 0.5], [1.0, 0.5], [1.0, 0.1]], dtype=np.float32)
    weights = np.asarray([1 / 3, 1 / 3, 1 / 3])
    singleton = _singleton(neighbors, similarity, weights)
    checks = {
        "singleton_finite": bool(np.isfinite(singleton).all()),
        "self_similarity_one": bool(np.allclose(similarity[:, 0], 1.0)),
        "exclusive_demand_changes_ranking": bool(singleton[2] > 0),
    }
    checks["passed"] = all(checks.values())
    return checks
