from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CostSelection:
    name: str
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    diagnostics: dict[str, Any]


def stable_ids(adapter: Any) -> np.ndarray:
    return np.asarray(
        [adapter.stable_id("static", index) for index in range(adapter.static_count)]
        + [adapter.stable_id("dynamic", index) for index in range(adapter.dynamic_count)]
    )


def _ordered(scores: np.ndarray, costs: np.ndarray, ids: np.ndarray, mode: str) -> np.ndarray:
    positive_zero = (costs == 0) & (scores > 0)
    zero_zero = (costs == 0) & (scores == 0)
    if mode == "ratio":
        ratio = np.divide(scores, costs, out=np.zeros_like(scores), where=costs > 0)
        classes = np.ones(scores.size, dtype=np.int8)
        classes[positive_zero] = 0
        classes[zero_zero] = 2
        return np.lexsort((ids, -ratio, classes))
    if mode == "score":
        return np.lexsort((ids, -scores))
    raise KeyError(mode)


def select_cost(
    adapter: Any,
    scores: np.ndarray,
    costs: np.ndarray,
    cap: float,
    name: str,
    mode: str = "ratio",
    header_cost: float = 0.0,
) -> CostSelection:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if values.shape != (adapter.total_count,) or item_costs.shape != values.shape:
        raise ValueError("scores and costs must match the reference model")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(item_costs).all()
        or np.any(values < 0)
        or np.any(item_costs < 0)
    ):
        raise ValueError("scores and costs must be finite and nonnegative")
    if header_cost > cap:
        raise ValueError("fixed header exceeds the registered cap")
    ids = stable_ids(adapter)
    order = _ordered(values, item_costs, ids, mode)
    remaining = float(cap - header_cost)
    selected = []
    for index in order:
        cost = float(item_costs[index])
        if cost <= remaining:
            selected.append(int(index))
            remaining -= cost
    greedy = np.asarray(selected, dtype=np.int64)
    feasible = np.flatnonzero(item_costs <= cap - header_cost)
    if feasible.size:
        best_order = np.lexsort((ids[feasible], -values[feasible]))
        best_single = feasible[best_order[0]]
        single_score = float(values[best_single])
    else:
        best_single = None
        single_score = float("-inf")
    greedy_score = float(values[greedy].sum())
    used_single = bool(best_single is not None and single_score > greedy_score)
    if used_single:
        greedy = np.asarray([best_single], dtype=np.int64)
        greedy_score = single_score
    greedy = np.sort(greedy)
    static = greedy[greedy < adapter.static_count]
    dynamic = greedy[greedy >= adapter.static_count] - adapter.static_count
    selected_cost = float(item_costs[greedy].sum())
    diagnostics = {
        "name": name,
        "mode": mode,
        "cap": float(cap),
        "header_cost": float(header_cost),
        "item_cap": float(cap - header_cost),
        "selected_item_cost": selected_cost,
        "estimated_total_cost": float(header_cost + selected_cost),
        "slack": float(cap - header_cost - selected_cost),
        "selected_total": int(greedy.size),
        "selected_static": int(static.size),
        "selected_dynamic": int(dynamic.size),
        "retained_e": greedy_score,
        "greedy_e": float(values[np.asarray(selected, dtype=np.int64)].sum()),
        "best_single_e": single_score if best_single is not None else None,
        "used_best_single": used_single,
        "zero_cost_positive_selected": int(
            np.count_nonzero((item_costs[greedy] == 0) & (values[greedy] > 0))
        ),
        "zero_zero_selected": int(
            np.count_nonzero((item_costs[greedy] == 0) & (values[greedy] == 0))
        ),
    }
    return CostSelection(name, static.astype(np.int64), dynamic.astype(np.int64), diagnostics)


def selection_from_indices(
    adapter: Any,
    scores: np.ndarray,
    static_indices: np.ndarray,
    dynamic_indices: np.ndarray,
    name: str,
) -> CostSelection:
    static = np.asarray(static_indices, dtype=np.int64)
    dynamic = np.asarray(dynamic_indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    return CostSelection(
        name,
        static,
        dynamic,
        {
            "name": name,
            "mode": "registered_p02",
            "selected_total": int(static.size + dynamic.size),
            "selected_static": int(static.size),
            "selected_dynamic": int(dynamic.size),
            "retained_e": float(
                values[static].sum() + values[adapter.static_count + dynamic].sum()
            ),
        },
    )


def decision_toy() -> dict[str, Any]:
    class Toy:
        static_count = 3
        dynamic_count = 0
        total_count = 3

        @staticmethod
        def stable_id(kind: str, row: int) -> str:
            return f"toy:{kind}:{row}"

    scores = np.asarray([10.0, 9.0, 8.0])
    unit = select_cost(Toy(), scores, np.ones(3), 2.0, "unit")
    unequal = select_cost(Toy(), scores, np.asarray([10.0, 1.0, 1.0]), 2.0, "unequal")
    zero = select_cost(Toy(), np.asarray([1.0, 0.0, 2.0]), np.asarray([0.0, 0.0, 2.0]), 2.0, "zero")
    header_blocked = False
    try:
        select_cost(Toy(), scores, np.ones(3), 0.5, "blocked", header_cost=1.0)
    except ValueError:
        header_blocked = True
    result = {
        "unit_ids": unit.static_indices.tolist(),
        "unequal_ids": unequal.static_indices.tolist(),
        "different_cost_changes_decision": bool(
            not np.array_equal(unit.static_indices, unequal.static_indices)
        ),
        "zero_cost_selection_terminates": bool(zero.static_indices.size <= 3),
        "positive_zero_selected": bool(0 in zero.static_indices),
        "header_over_cap_blocked": header_blocked,
    }
    result["passed"] = all(
        value for key, value in result.items() if isinstance(value, bool)
    )
    return result
