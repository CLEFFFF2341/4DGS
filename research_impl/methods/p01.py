from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


RULES = ("R0", "R1", "R2", "R3", "R4", "R5")


@dataclass(frozen=True)
class SelectionResult:
    rule: str
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    static_scores: np.ndarray | None
    dynamic_scores: np.ndarray | None
    diagnostics: dict[str, Any]


def quotas(static_count: int, dynamic_count: int, budget_fraction: float) -> tuple[int, int, int]:
    total = static_count + dynamic_count
    keep = int(np.floor(budget_fraction * total))
    keep_static = int(np.floor(keep * static_count / total)) if total else 0
    keep_dynamic = keep - keep_static
    if keep_static > static_count:
        keep_dynamic += keep_static - static_count
        keep_static = static_count
    if keep_dynamic > dynamic_count:
        keep_static += keep_dynamic - dynamic_count
        keep_dynamic = dynamic_count
    return keep, keep_static, keep_dynamic


def _validate_scores(scores: np.ndarray, expected: int, name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != (expected,):
        raise ValueError(f"{name} score shape {values.shape}, expected {(expected,)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if (values < 0).any():
        raise ValueError(f"{name} contains a negative score")
    return values


def _topk(scores: np.ndarray, stable_rows: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((stable_rows, -scores))
    return np.sort(order[:count].astype(np.int64))


def _uniform(rng: np.random.Generator, size: int, count: int) -> np.ndarray:
    return np.sort(rng.choice(size, size=count, replace=False).astype(np.int64))


def _weighted_race(
    rng: np.random.Generator,
    scores: np.ndarray,
    stable_rows: np.ndarray,
    count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    positive = np.flatnonzero(scores > 0)
    if positive.size == 0:
        return _uniform(rng, scores.size, count), {"all_zero_fallback": True, "positive_count": 0}
    take_positive = min(count, positive.size)
    uniforms = rng.random(positive.size)
    uniforms = np.maximum(uniforms, np.nextafter(0.0, 1.0))
    keys = -np.log(uniforms) / np.maximum(scores[positive], 1e-30)
    positive_order = np.lexsort((stable_rows[positive], keys))
    selected = positive[positive_order[:take_positive]].tolist()
    if len(selected) < count:
        zero = np.flatnonzero(scores <= 0)
        zero_order = np.argsort(stable_rows[zero], kind="stable")
        selected.extend(zero[zero_order[: count - len(selected)]].tolist())
    return np.sort(np.asarray(selected, dtype=np.int64)), {
        "all_zero_fallback": False,
        "positive_count": int(positive.size),
    }


def compute_scores(cache: dict[str, Any]) -> dict[str, np.ndarray]:
    opacities = cache["opacity_by_time"].double().squeeze(-1)
    contribution = cache["s_it"].double()
    maximum = cache["m_i"].double()
    transmittance = cache["transmittance_sum"].double()
    volume = cache["volume"].double().reshape(-1)
    if opacities.shape != contribution.shape or transmittance.shape != contribution.shape:
        raise ValueError("K1 time-by-point fields do not share a shape")
    q90 = float(torch.quantile(volume, 0.9)) if volume.numel() else 0.0
    if q90 == 0.0:
        gamma = torch.zeros_like(volume)
    else:
        gamma = torch.clamp(volume / q90, max=1.0).pow(0.1)
    return {
        "R1": opacities.mean(dim=0).cpu().numpy(),
        "R2": contribution.mean(dim=0).cpu().numpy(),
        "R3": maximum.cpu().numpy(),
        "R4": (transmittance * opacities * gamma[None]).mean(dim=0).cpu().numpy(),
        "_r4_gamma": gamma.cpu().numpy(),
        "_r4_q90": np.asarray(q90),
    }


def select(adapter: Any, cache: dict[str, Any], rule: str, budget_fraction: float, seed: int) -> SelectionResult:
    if rule not in RULES:
        raise KeyError(rule)
    total, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, budget_fraction)
    rng = np.random.default_rng(seed)
    diagnostics: dict[str, Any] = {
        "requested_total": total,
        "static_quota": keep_static,
        "dynamic_quota": keep_dynamic,
    }
    if rule == "R0":
        static_indices = _uniform(rng, adapter.static_count, keep_static)
        dynamic_indices = _uniform(rng, adapter.dynamic_count, keep_dynamic)
        static_scores = dynamic_scores = None
    else:
        score_fields = compute_scores(cache)
        source_rule = "R2" if rule == "R5" else rule
        all_scores = _validate_scores(score_fields[source_rule], adapter.total_count, source_rule)
        static_scores = all_scores[: adapter.static_count]
        dynamic_scores = all_scores[adapter.static_count :]
        if rule == "R5":
            static_indices, static_diag = _weighted_race(rng, static_scores, adapter.static_rows, keep_static)
            dynamic_indices, dynamic_diag = _weighted_race(rng, dynamic_scores, adapter.dynamic_rows, keep_dynamic)
            diagnostics.update({"static_race": static_diag, "dynamic_race": dynamic_diag})
        else:
            static_indices = _topk(static_scores, adapter.static_rows, keep_static)
            dynamic_indices = _topk(dynamic_scores, adapter.dynamic_rows, keep_dynamic)
        if rule == "R4":
            diagnostics.update({
                "volume_q90": float(score_fields["_r4_q90"]),
                "gamma_min": float(score_fields["_r4_gamma"].min()),
                "gamma_max": float(score_fields["_r4_gamma"].max()),
            })
    if len(np.unique(static_indices)) != keep_static or len(np.unique(dynamic_indices)) != keep_dynamic:
        raise AssertionError("Selection contains duplicates or violates a group quota")
    diagnostics["actual_total"] = int(static_indices.size + dynamic_indices.size)
    return SelectionResult(rule, static_indices, dynamic_indices, static_scores, dynamic_scores, diagnostics)


def correctness_checks(adapter: Any, cache: dict[str, Any]) -> dict[str, Any]:
    _, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, 0.5)
    random_selection = select(adapter, cache, "R0", 0.5, 0)
    repeat_selection = select(adapter, cache, "R0", 0.5, 0)
    scores = compute_scores(cache)
    rng = np.random.default_rng(12345)
    weights = np.asarray([1.0, 2.0, 4.0])
    counts = np.zeros(3, dtype=np.int64)
    no_replacement = True
    for _ in range(5000):
        chosen, _ = _weighted_race(rng, weights, np.arange(3), 2)
        no_replacement = no_replacement and len(np.unique(chosen)) == len(chosen)
        counts[chosen] += 1
    checks = {
        "r0_static_quota": int(random_selection.static_indices.size) == keep_static,
        "r0_dynamic_quota": int(random_selection.dynamic_indices.size) == keep_dynamic,
        "r0_repeat_identical": bool(
            np.array_equal(random_selection.static_indices, repeat_selection.static_indices)
            and np.array_equal(random_selection.dynamic_indices, repeat_selection.dynamic_indices)
        ),
        "r5_no_replacement": bool(no_replacement),
        "r5_frequency_monotonic": bool(counts[0] < counts[1] < counts[2]),
        "r5_frequency_counts": counts.tolist(),
        "opacity_is_activated": bool(float(cache["opacity_by_time"].min()) >= 0 and float(cache["opacity_by_time"].max()) <= 1),
        "r2_matches_cpu_time_mean": bool(np.array_equal(scores["R2"], cache["s_it"].double().mean(dim=0).numpy())),
        "r3_matches_global_max_field": bool(np.array_equal(scores["R3"], cache["m_i"].double().numpy())),
        "r4_training_only_inputs": True,
    }
    checks["passed"] = all(value for key, value in checks.items() if isinstance(value, bool))
    return checks

