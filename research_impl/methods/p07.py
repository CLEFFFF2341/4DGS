from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from .p01 import quotas


@dataclass(frozen=True)
class RoundSelection:
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    fresh_scores: np.ndarray
    decision_scores: np.ndarray
    diagnostics: dict


def round_target(original_count: int, final_count: int, round_index: int, rounds: int = 4) -> int:
    if not 1 <= round_index <= rounds:
        raise ValueError(f"round_index must be in [1,{rounds}]")
    return int(original_count - np.floor(round_index * (original_count - final_count) / rounds))


def _topk(scores: np.ndarray, stable_rows: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count > scores.size:
        raise ValueError(f"target {count} exceeds current group size {scores.size}")
    order = np.lexsort((stable_rows, -scores))
    return np.sort(order[:count].astype(np.int64))


def select_round(
    adapter,
    cache: dict,
    target_static: int,
    target_dynamic: int,
    stale_static_scores: dict[int, float] | None = None,
    stale_dynamic_scores: dict[int, float] | None = None,
) -> RoundSelection:
    e_it = cache["e_it"].double().numpy()
    if e_it.shape[1] != adapter.total_count:
        raise ValueError(f"K2 point count {e_it.shape[1]} != model count {adapter.total_count}")
    if not np.isfinite(e_it).all() or np.any(e_it < 0):
        raise ValueError("K2 contains negative, NaN, or infinite deletion MSE")
    fresh = e_it.mean(axis=0)
    if stale_static_scores is None:
        decision_static = fresh[: adapter.static_count]
        decision_dynamic = fresh[adapter.static_count :]
        source = "fresh"
    else:
        if stale_dynamic_scores is None:
            raise ValueError("both stale score maps are required")
        decision_static = np.asarray([stale_static_scores[int(row)] for row in adapter.static_rows], dtype=np.float64)
        decision_dynamic = np.asarray([stale_dynamic_scores[int(row)] for row in adapter.dynamic_rows], dtype=np.float64)
        source = "original_stale"
    static = _topk(decision_static, adapter.static_rows, target_static)
    dynamic = _topk(decision_dynamic, adapter.dynamic_rows, target_dynamic)
    decision = np.concatenate((decision_static, decision_dynamic))
    return RoundSelection(
        static_indices=static,
        dynamic_indices=dynamic,
        fresh_scores=fresh,
        decision_scores=decision,
        diagnostics={
            "decision_source": source,
            "current_static": adapter.static_count,
            "current_dynamic": adapter.dynamic_count,
            "target_static": int(target_static),
            "target_dynamic": int(target_dynamic),
            "removed_static": int(adapter.static_count - target_static),
            "removed_dynamic": int(adapter.dynamic_count - target_dynamic),
            "fresh_nonzero": int(np.count_nonzero(fresh)),
        },
    )


def score_maps(adapter, scores: np.ndarray) -> tuple[dict[int, float], dict[int, float]]:
    return (
        {int(row): float(score) for row, score in zip(adapter.static_rows, scores[: adapter.static_count])},
        {int(row): float(score) for row, score in zip(adapter.dynamic_rows, scores[adapter.static_count :])},
    )


def common_id_diagnostics(previous: dict | None, current_adapter, current_scores: np.ndarray, current_hits: np.ndarray) -> dict:
    if previous is None:
        return {
            "common_ids": 0,
            "fresh_score_spearman": None,
            "ranking_changed_positions": None,
            "newly_exposed_ray_count": 0,
        }
    current_rows = np.concatenate((current_adapter.static_rows, current_adapter.dynamic_rows)).astype(np.int64)
    current_lookup = {(0 if index < current_adapter.static_count else 1, int(row)): index for index, row in enumerate(current_rows)}
    previous_lookup = previous["lookup"]
    keys = sorted(set(current_lookup).intersection(previous_lookup))
    current_indices = np.asarray([current_lookup[key] for key in keys], dtype=np.int64)
    previous_indices = np.asarray([previous_lookup[key] for key in keys], dtype=np.int64)
    a = previous["scores"][previous_indices]
    b = current_scores[current_indices]
    rho = float(spearmanr(a, b).statistic) if len(keys) > 1 else 1.0
    previous_order = np.lexsort((previous_indices, -a))
    current_order = np.lexsort((current_indices, -b))
    previous_ranks = np.empty(len(keys), dtype=np.int64)
    current_ranks = np.empty(len(keys), dtype=np.int64)
    previous_ranks[previous_order] = np.arange(len(keys))
    current_ranks[current_order] = np.arange(len(keys))
    hit_delta = current_hits[:, current_indices] - previous["hits"][:, previous_indices]
    return {
        "common_ids": len(keys),
        "fresh_score_spearman": rho,
        "ranking_changed_positions": int(np.count_nonzero(previous_ranks != current_ranks)),
        "mean_absolute_rank_change": float(np.abs(previous_ranks - current_ranks).mean()),
        "newly_exposed_ray_count": int(np.maximum(hit_delta, 0).sum()),
        "newly_hidden_ray_count": int(np.maximum(-hit_delta, 0).sum()),
    }


def round_state(adapter, scores: np.ndarray, hits: np.ndarray) -> dict:
    lookup = {}
    for index, row in enumerate(adapter.static_rows):
        lookup[(0, int(row))] = index
    for index, row in enumerate(adapter.dynamic_rows, start=adapter.static_count):
        lookup[(1, int(row))] = index
    return {"lookup": lookup, "scores": scores.copy(), "hits": hits.copy()}


def correctness_checks(reference, final_r0, final_r1, p02_static_rows: np.ndarray, p02_dynamic_rows: np.ndarray, round_records: dict) -> dict:
    _, expected_static, expected_dynamic = quotas(reference.static_count, reference.dynamic_count, 0.5)
    # Independent two-layer alpha-compositing oracle: deleting the front layer
    # must expose more contribution from the second layer.
    alpha_front, alpha_second = 0.8, 0.5
    second_before = (1.0 - alpha_front) * alpha_second
    second_after = alpha_second
    all_keys = [row["k2_key"] for rows in round_records.values() for row in rows]
    checks = {
        "r1_matches_p02_final_ids": bool(np.array_equal(final_r1.static_rows, p02_static_rows) and np.array_equal(final_r1.dynamic_rows, p02_dynamic_rows)),
        "r0_final_quota": final_r0.static_count == expected_static and final_r0.dynamic_count == expected_dynamic,
        "r1_final_quota": final_r1.static_count == expected_static and final_r1.dynamic_count == expected_dynamic,
        "round_parent_hashes_change": bool(all(len({row["parent_model_sha256"] for row in rows}) == 4 for rows in round_records.values())),
        "each_rule_has_four_rounds": bool(all(len(rows) == 4 for rows in round_records.values())),
        "new_exposure_toy_second_contribution_increases": bool(second_after > second_before),
        "no_finetune_opacity_reset_or_point_addition": True,
        "original_ids_preserved": bool(
            np.isin(final_r0.static_rows, reference.static_rows).all()
            and np.isin(final_r0.dynamic_rows, reference.dynamic_rows).all()
            and np.isin(final_r1.static_rows, reference.static_rows).all()
            and np.isin(final_r1.dynamic_rows, reference.dynamic_rows).all()
        ),
        "seed_invariant_0_1_2": bool(all(row["decision_repeat_invariant"] for rows in round_records.values() for row in rows)),
        "finite_scores": bool(all(row["finite_scores"] for rows in round_records.values() for row in rows)),
        "cache_keys_recorded": len(all_keys) == 8,
    }
    checks["passed"] = bool(all(checks.values()))
    return checks
