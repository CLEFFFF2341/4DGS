from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemporalPlan:
    active: np.ndarray
    sample_times: np.ndarray
    bin_for_time: np.ndarray
    bin_lengths: np.ndarray
    gamma: float
    lambdas: dict[str, float]
    diagnostics: dict[str, Any]
    traces: dict[str, list[dict[str, Any]]]


def temporal_bins(sample_times: np.ndarray, duration: int = 300) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(sample_times, dtype=np.int64)
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("sample times must be a non-empty strictly increasing vector")
    frames = np.arange(duration, dtype=np.int64)
    distances = np.abs(frames[:, None] - times[None, :])
    # np.argmin returns the first index, so exact midpoint ties go to the earlier bin.
    mapping = np.argmin(distances, axis=1).astype(np.int64)
    lengths = np.bincount(mapping, minlength=times.size).astype(np.int64)
    if np.any(lengths == 0):
        raise ValueError("every temporal sample must own at least one integer frame")
    return mapping, lengths


def _prefer_second(
    cost_first: np.ndarray,
    switches_first: np.ndarray,
    cost_second: np.ndarray,
    switches_second: np.ndarray,
) -> np.ndarray:
    lower_cost = cost_second < cost_first
    tied_cost = cost_second == cost_first
    fewer_switches = switches_second < switches_first
    # When both cost and switch count tie, predecessor state 0 (the first candidate)
    # wins. This supplies the registered inactive-first stable tie break.
    return lower_cost | (tied_cost & fewer_switches)


def solve_chain(
    errors: np.ndarray,
    bin_lengths: np.ndarray,
    lambda_value: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(errors, dtype=np.float64)
    lengths = np.asarray(bin_lengths, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != lengths.size:
        raise ValueError("errors must have shape [points, bins]")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("errors must be finite and nonnegative")
    if not np.isfinite(lambda_value) or not np.isfinite(gamma) or lambda_value < 0 or gamma < 0:
        raise ValueError("lambda and gamma must be finite and nonnegative")
    count, bins = values.shape
    if count == 0:
        return (
            np.empty((0, bins), dtype=bool),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
        )
    inactive_cost = lengths[0] * values[:, 0]
    active_cost = np.full(count, lambda_value * lengths[0], dtype=np.float64)
    inactive_switches = np.zeros(count, dtype=np.int32)
    active_switches = np.zeros(count, dtype=np.int32)
    parent_inactive = np.zeros((bins, count), dtype=bool)
    parent_active = np.zeros((bins, count), dtype=bool)
    for index in range(1, bins):
        to_inactive_from_inactive = inactive_cost
        to_inactive_from_active = active_cost + gamma
        switches_ii = inactive_switches
        switches_ai = active_switches + 1
        choose_active_parent = _prefer_second(
            to_inactive_from_inactive,
            switches_ii,
            to_inactive_from_active,
            switches_ai,
        )
        next_inactive_cost = np.where(
            choose_active_parent, to_inactive_from_active, to_inactive_from_inactive
        ) + lengths[index] * values[:, index]
        next_inactive_switches = np.where(
            choose_active_parent, switches_ai, switches_ii
        ).astype(np.int32)

        to_active_from_inactive = inactive_cost + gamma
        to_active_from_active = active_cost
        switches_ia = inactive_switches + 1
        switches_aa = active_switches
        choose_active_parent_for_active = _prefer_second(
            to_active_from_inactive,
            switches_ia,
            to_active_from_active,
            switches_aa,
        )
        next_active_cost = np.where(
            choose_active_parent_for_active, to_active_from_active, to_active_from_inactive
        ) + lambda_value * lengths[index]
        next_active_switches = np.where(
            choose_active_parent_for_active, switches_aa, switches_ia
        ).astype(np.int32)

        parent_inactive[index] = choose_active_parent
        parent_active[index] = choose_active_parent_for_active
        inactive_cost, active_cost = next_inactive_cost, next_active_cost
        inactive_switches, active_switches = next_inactive_switches, next_active_switches

    # Exact terminal ties prefer the inactive state, before considering switch count.
    state = active_cost < inactive_cost
    objective = np.where(state, active_cost, inactive_cost)
    switches = np.where(state, active_switches, inactive_switches).astype(np.int32)
    active = np.zeros((count, bins), dtype=bool)
    active[:, -1] = state
    rows = np.arange(count)
    for index in range(bins - 1, 0, -1):
        state = np.where(
            state,
            parent_active[index, rows],
            parent_inactive[index, rows],
        )
        active[:, index - 1] = state
    return active, objective, switches


def _selection_stats(active: np.ndarray, errors: np.ndarray, lengths: np.ndarray) -> dict[str, Any]:
    active_frames = int(np.sum(active * lengths[None, :], dtype=np.int64))
    proxy_distortion = float(np.sum((~active) * errors * lengths[None, :], dtype=np.float64))
    switches = int(np.count_nonzero(active[:, 1:] != active[:, :-1]))
    return {
        "active_frames": active_frames,
        "proxy_distortion": proxy_distortion,
        "switches": switches,
    }


def select_group(
    errors: np.ndarray,
    bin_lengths: np.ndarray,
    gamma: float,
    cap: int,
    iterations: int = 20,
) -> tuple[np.ndarray, float, list[dict[str, Any]], dict[str, Any]]:
    values = np.asarray(errors, dtype=np.float64)
    lengths = np.asarray(bin_lengths, dtype=np.int64)
    if values.shape[0] == 0:
        empty = np.empty_like(values, dtype=bool)
        return empty, 0.0, [], {"active_frames": 0, "proxy_distortion": 0.0, "switches": 0}
    lower = 0.0
    upper = float(values.max() + 2.0 * gamma / int(lengths.min()))
    active, _, _ = solve_chain(values, lengths, upper, gamma)
    best_stats = _selection_stats(active, values, lengths)
    if best_stats["active_frames"] > cap:
        raise RuntimeError("registered lambda upper bound did not yield a feasible solution")
    best = active
    best_lambda = upper
    trace = [
        {
            "iteration": -1,
            "lambda": upper,
            "feasible": True,
            **best_stats,
        }
    ]
    for iteration in range(iterations):
        midpoint = 0.5 * (lower + upper)
        candidate, _, _ = solve_chain(values, lengths, midpoint, gamma)
        stats = _selection_stats(candidate, values, lengths)
        feasible = stats["active_frames"] <= cap
        trace.append(
            {
                "iteration": iteration,
                "lambda": midpoint,
                "feasible": bool(feasible),
                **stats,
            }
        )
        if feasible:
            if (
                stats["active_frames"] > best_stats["active_frames"]
                or (
                    stats["active_frames"] == best_stats["active_frames"]
                    and stats["proxy_distortion"] < best_stats["proxy_distortion"]
                )
            ):
                best = candidate
                best_lambda = midpoint
                best_stats = stats
            upper = midpoint
        else:
            lower = midpoint
    return best, best_lambda, trace, best_stats


def build_plan(
    e_it: np.ndarray,
    sample_times: np.ndarray,
    static_count: int,
    dynamic_count: int,
    rule: str,
    duration: int = 300,
) -> TemporalPlan:
    if rule not in {"R0", "R1"}:
        raise KeyError(rule)
    errors = np.asarray(e_it, dtype=np.float64)
    if errors.shape != (sample_times.size, static_count + dynamic_count):
        raise ValueError("K2 e_it shape does not match the reference model")
    mapping, lengths = temporal_bins(sample_times, duration)
    point_bin_errors = errors.T
    positive_weighted = (point_bin_errors * lengths[None, :])
    positive_weighted = positive_weighted[positive_weighted > 0]
    gamma_base = 0.1 * float(np.median(positive_weighted)) if positive_weighted.size else 0.0
    gamma = gamma_base if rule == "R0" else 0.0
    groups = {}
    offset = 0
    for kind, count in (("static", static_count), ("dynamic", dynamic_count)):
        cap = int(np.floor(0.5 * count * duration))
        selected, lambda_value, trace, stats = select_group(
            point_bin_errors[offset : offset + count], lengths, gamma, cap
        )
        groups[kind] = {
            "active": selected,
            "lambda": lambda_value,
            "trace": trace,
            "stats": {
                **stats,
                "cap": cap,
                "budget_utilization": stats["active_frames"] / cap if cap else 1.0,
                "budget_gap": bool(cap > 0 and stats["active_frames"] < 0.95 * cap),
                "unique_active_min": int(selected.sum(axis=0).min()) if count else 0,
                "unique_active_max": int(selected.sum(axis=0).max()) if count else 0,
                "entities_ever_active": int(np.any(selected, axis=1).sum()),
            },
        }
        offset += count
    active = np.concatenate((groups["static"]["active"], groups["dynamic"]["active"]), axis=0)
    total_cap = int(np.floor(0.5 * (static_count + dynamic_count) * duration))
    actual = int(np.sum(active * lengths[None, :], dtype=np.int64))
    diagnostics = {
        "rule": rule,
        "gamma": gamma,
        "gamma_base": gamma_base,
        "total_cap": total_cap,
        "actual_active_frames": actual,
        "budget_utilization": actual / total_cap,
        "budget_gap": bool(actual < 0.95 * total_cap),
        "loaded_entities": static_count + dynamic_count,
        "bins": int(sample_times.size),
        "bin_lengths": lengths.tolist(),
        "static": groups["static"]["stats"],
        "dynamic": groups["dynamic"]["stats"],
    }
    return TemporalPlan(
        active=active,
        sample_times=np.asarray(sample_times, dtype=np.int64),
        bin_for_time=mapping,
        bin_lengths=lengths,
        gamma=gamma,
        lambdas={kind: float(groups[kind]["lambda"]) for kind in ("static", "dynamic")},
        diagnostics=diagnostics,
        traces={kind: groups[kind]["trace"] for kind in ("static", "dynamic")},
    )


def active_indices(plan: TemporalPlan, timestamp: int, static_count: int) -> tuple[np.ndarray, np.ndarray]:
    bin_index = int(plan.bin_for_time[int(timestamp)])
    mask = plan.active[:, bin_index]
    return np.flatnonzero(mask[:static_count]), np.flatnonzero(mask[static_count:])


def rle_intervals(plan: TemporalPlan) -> dict[str, np.ndarray]:
    padded = np.pad(plan.active.astype(np.int8), ((0, 0), (1, 1)))
    changes = np.diff(padded, axis=1)
    start_point, start_bin = np.nonzero(changes == 1)
    end_point, end_exclusive = np.nonzero(changes == -1)
    if not np.array_equal(start_point, end_point):
        raise AssertionError("RLE boundary pairing failed")
    end_bin = end_exclusive - 1
    bin_starts = np.asarray(
        [np.flatnonzero(plan.bin_for_time == index)[0] for index in range(plan.sample_times.size)],
        dtype=np.int64,
    )
    bin_ends = np.asarray(
        [np.flatnonzero(plan.bin_for_time == index)[-1] for index in range(plan.sample_times.size)],
        dtype=np.int64,
    )
    return {
        "point": start_point.astype(np.int64),
        "start_bin": start_bin.astype(np.int64),
        "end_bin": end_bin.astype(np.int64),
        "start_time": bin_starts[start_bin],
        "end_time": bin_ends[end_bin],
    }


def reconstruct_rle(count: int, bins: int, intervals: dict[str, np.ndarray]) -> np.ndarray:
    differences = np.zeros((count, bins + 1), dtype=np.int16)
    np.add.at(differences, (intervals["point"], intervals["start_bin"]), 1)
    np.add.at(differences, (intervals["point"], intervals["end_bin"] + 1), -1)
    return np.cumsum(differences[:, :bins], axis=1) > 0


def _objective(sequence: np.ndarray, errors: np.ndarray, lengths: np.ndarray, lambda_value: float, gamma: float) -> float:
    return float(
        np.sum(lengths * errors * (~sequence))
        + lambda_value * np.sum(lengths * sequence)
        + gamma * np.count_nonzero(sequence[1:] != sequence[:-1])
    )


def dp_toy() -> dict[str, Any]:
    rng = np.random.default_rng(18)
    cases = []
    for points in range(1, 4):
        for bins in range(1, 6):
            errors = rng.integers(0, 5, size=(points, bins)).astype(np.float64) / 10.0
            lengths = rng.integers(1, 4, size=bins).astype(np.int64)
            lambda_value = 0.17
            gamma = 0.09
            active, objectives, _ = solve_chain(errors, lengths, lambda_value, gamma)
            for row in range(points):
                brute = min(
                    _objective(np.asarray(sequence, dtype=bool), errors[row], lengths, lambda_value, gamma)
                    for sequence in itertools.product((False, True), repeat=bins)
                )
                cases.append(abs(float(objectives[row]) - brute))
    empty, _, _ = solve_chain(np.empty((0, 3)), np.ones(3, dtype=np.int64), 0.0, 0.0)
    zeros, _, _ = solve_chain(np.zeros((2, 4)), np.ones(4, dtype=np.int64), 0.0, 0.0)
    result = {
        "cases": len(cases),
        "maximum_objective_error": max(cases),
        "empty_group_shape": list(empty.shape),
        "all_zero_inactive": bool(not zeros.any()),
    }
    result["passed"] = bool(
        result["maximum_objective_error"] <= 1e-12
        and result["empty_group_shape"] == [0, 3]
        and result["all_zero_inactive"]
    )
    return result
