from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.sh_utils import C0

from .p01 import quotas


@dataclass(frozen=True)
class PairPlan:
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    static_pairs: np.ndarray
    dynamic_pairs: np.ndarray
    static_weights: np.ndarray
    dynamic_weights: np.ndarray
    diagnostics: dict[str, Any]


def _unique_edges(neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = neighbors.shape[0]
    source = np.repeat(np.arange(count, dtype=np.int64), neighbors.shape[1])
    target = neighbors.reshape(-1).astype(np.int64)
    valid = (target >= 0) & (target < count) & (source != target)
    lower = np.minimum(source[valid], target[valid])
    upper = np.maximum(source[valid], target[valid])
    keys = lower.astype(np.uint64) * np.uint64(count) + upper.astype(np.uint64)
    keys = np.unique(keys)
    return (keys // np.uint64(count)).astype(np.int64), (keys % np.uint64(count)).astype(np.int64)


def _filter_edges(
    positions: torch.Tensor,
    colors: np.ndarray,
    mean_scales: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    length_scale: float,
    opacity_envelopes: torch.Tensor | None,
    batch_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    kept_lower = []
    kept_upper = []
    kept_distance = []
    passed_color_position = 0
    for start in range(0, lower.size, batch_size):
        stop = min(start + batch_size, lower.size)
        left = lower[start:stop]
        right = upper[start:stop]
        left_t = torch.from_numpy(left)
        right_t = torch.from_numpy(right)
        max_position = torch.linalg.vector_norm(
            positions.index_select(1, left_t) - positions.index_select(1, right_t), dim=2
        ).amax(dim=0).numpy()
        color_distance = np.linalg.norm(colors[left] - colors[right], axis=1)
        scale_limit = 2.0 * np.maximum(mean_scales[left], mean_scales[right])
        mask = (color_distance <= 0.1) & (max_position <= scale_limit)
        passed_color_position += int(mask.sum())
        if opacity_envelopes is not None and np.any(mask):
            candidate_left = torch.from_numpy(left[mask])
            candidate_right = torch.from_numpy(right[mask])
            envelope_difference = (
                opacity_envelopes.index_select(1, candidate_left)
                - opacity_envelopes.index_select(1, candidate_right)
            ).abs().amax(dim=0).numpy()
            local = np.flatnonzero(mask)
            mask[local] &= envelope_difference <= 0.1
        if np.any(mask):
            kept_lower.append(left[mask])
            kept_upper.append(right[mask])
            kept_distance.append(max_position[mask] / length_scale + color_distance[mask])
    if not kept_lower:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_i.copy(), empty_f, {
            "input_edges": int(lower.size),
            "passed_color_position": passed_color_position,
            "feasible_edges": 0,
        }
    result_lower = np.concatenate(kept_lower)
    result_upper = np.concatenate(kept_upper)
    result_distance = np.concatenate(kept_distance).astype(np.float64)
    return result_lower, result_upper, result_distance, {
        "input_edges": int(lower.size),
        "passed_color_position": passed_color_position,
        "feasible_edges": int(result_lower.size),
    }


def _greedy_pairs(
    lower: np.ndarray,
    upper: np.ndarray,
    distance: np.ndarray,
    scores: np.ndarray,
    target_merges: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((upper, lower, distance))
    used = np.zeros(scores.size, dtype=bool)
    pairs = []
    weights = []
    for index in order:
        left = int(lower[index])
        right = int(upper[index])
        if used[left] or used[right]:
            continue
        if scores[left] > scores[right] or (scores[left] == scores[right] and left < right):
            parent, child = left, right
        else:
            parent, child = right, left
        denominator = scores[left] + scores[right]
        left_weight = scores[left] / denominator if denominator > 0 else 0.5
        pairs.append((parent, child, left, right, float(distance[index])))
        weights.append(left_weight)
        used[left] = True
        used[right] = True
        if len(pairs) >= target_merges:
            break
    if not pairs:
        return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.asarray(pairs, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _finalize_group(
    count: int,
    keep: int,
    scores: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.ones(count, dtype=bool)
    entity_scores = scores.astype(np.float64).copy()
    parent_to_pair = {}
    for pair_index, pair in enumerate(pairs):
        parent, child = int(pair[0]), int(pair[1])
        active[child] = False
        entity_scores[parent] = scores[parent] + scores[child]
        parent_to_pair[parent] = pair_index
    candidates = np.flatnonzero(active)
    order = np.lexsort((candidates, -entity_scores[candidates]))
    selected = np.sort(candidates[order[:keep]].astype(np.int64))
    retained_pairs = np.asarray(
        [parent_to_pair[index] for index in selected if index in parent_to_pair], dtype=np.int64
    )
    return selected, retained_pairs


@torch.no_grad()
def build_pair_plan(adapter: Any, k0: dict[str, Any], k4: dict[str, Any], scores: np.ndarray) -> PairPlan:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != (adapter.total_count,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("P02.E must be finite, nonnegative, and match the model")
    length_scale = float(k4["length_scale"])
    if length_scale == 0:
        length_scale = 1.0
    colors = (C0 * k0["features"][:, 0, :] + 0.5).double().numpy()
    mean_scales = k0["scaling"].double().mean(dim=1).numpy()
    positions = k0["positions"].float()
    _, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, 0.5)
    envelope_dynamic = torch.stack(
        [adapter.model.get_opacity_at_t(timestamp, mode=2, training=False).squeeze(-1).cpu() for timestamp in range(300)],
        dim=0,
    )
    group_results = {}
    offset = 0
    for kind, count, keep, neighbor_key, envelope in (
        ("static", adapter.static_count, keep_static, "static_neighbors", None),
        ("dynamic", adapter.dynamic_count, keep_dynamic, "dynamic_neighbors", envelope_dynamic),
    ):
        neighbors = k4[neighbor_key].numpy()
        lower, upper = _unique_edges(neighbors)
        filtered_lower, filtered_upper, distances, filtering = _filter_edges(
            positions[:, offset : offset + count],
            colors[offset : offset + count],
            mean_scales[offset : offset + count],
            lower,
            upper,
            length_scale,
            envelope,
        )
        pairs, weights = _greedy_pairs(
            filtered_lower,
            filtered_upper,
            distances,
            values[offset : offset + count],
            count - keep,
        )
        selected, retained_pair_indices = _finalize_group(
            count, keep, values[offset : offset + count], pairs
        )
        group_results[kind] = {
            "selected": selected,
            "pairs": pairs,
            "weights": weights,
            "retained_pair_indices": retained_pair_indices,
            "diagnostics": {
                **filtering,
                "point_count": count,
                "target_keep": keep,
                "target_merges": count - keep,
                "accepted_disjoint_pairs": int(pairs.shape[0]),
                "retained_merged_parents": int(retained_pair_indices.size),
                "fallback_deletions": int(max(0, count - pairs.shape[0] - keep)),
                "opportunity_fraction": float(pairs.shape[0] / count),
            },
        }
        offset += count
    diagnostics = {
        "length_scale": length_scale,
        "static": group_results["static"]["diagnostics"],
        "dynamic": group_results["dynamic"]["diagnostics"],
        "actual_total": int(group_results["static"]["selected"].size + group_results["dynamic"]["selected"].size),
    }
    diagnostics["opportunity_insufficient"] = bool(
        (group_results["static"]["pairs"].shape[0] + group_results["dynamic"]["pairs"].shape[0])
        < 0.01 * adapter.total_count
    )
    return PairPlan(
        static_indices=group_results["static"]["selected"],
        dynamic_indices=group_results["dynamic"]["selected"],
        static_pairs=group_results["static"]["pairs"],
        dynamic_pairs=group_results["dynamic"]["pairs"],
        static_weights=group_results["static"]["weights"],
        dynamic_weights=group_results["dynamic"]["weights"],
        diagnostics=diagnostics,
    )


def _rotation_matrices(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    xyzw = quaternion[..., [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(quaternion.shape[:-1] + (3, 3))


def _moment_shape(
    positions_left: np.ndarray,
    positions_right: np.ndarray,
    scales_left: np.ndarray,
    scales_right: np.ndarray,
    rotations_left: np.ndarray,
    rotations_right: np.ndarray,
    left_weight: float,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right_weight = 1.0 - left_weight
    mean = left_weight * positions_left + right_weight * positions_right
    rotation_left = _rotation_matrices(rotations_left)
    rotation_right = _rotation_matrices(rotations_right)
    diagonal_left = np.diag(np.square(scales_left))
    diagonal_right = np.diag(np.square(scales_right))
    covariance_left = rotation_left @ diagonal_left @ np.swapaxes(rotation_left, -1, -2)
    covariance_right = rotation_right @ diagonal_right @ np.swapaxes(rotation_right, -1, -2)
    delta_left = positions_left - mean
    delta_right = positions_right - mean
    covariance = (
        left_weight * (covariance_left + delta_left[..., :, None] * delta_left[..., None, :])
        + right_weight * (covariance_right + delta_right[..., :, None] * delta_right[..., None, :])
    ).mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8 * length_scale * length_scale)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    quaternion_xyzw = Rotation.from_matrix(eigenvectors).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    if quaternion_wxyz[0] < 0:
        quaternion_wxyz *= -1
    return mean, np.sqrt(eigenvalues), quaternion_wxyz


def _moment_shape_batch(
    positions_left: np.ndarray,
    positions_right: np.ndarray,
    scales_left: np.ndarray,
    scales_right: np.ndarray,
    rotations_left: np.ndarray,
    rotations_right: np.ndarray,
    left_weights: np.ndarray,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = left_weights[None, :, None]
    mean = weights * positions_left + (1.0 - weights) * positions_right
    rotation_left = _rotation_matrices(rotations_left)
    rotation_right = _rotation_matrices(rotations_right)
    scale2_left = np.square(scales_left)[None, :, None, :]
    scale2_right = np.square(scales_right)[None, :, None, :]
    covariance_left = (rotation_left * scale2_left) @ np.swapaxes(rotation_left, -1, -2)
    covariance_right = (rotation_right * scale2_right) @ np.swapaxes(rotation_right, -1, -2)
    delta_left = positions_left - mean
    delta_right = positions_right - mean
    matrix_weights = left_weights[None, :, None, None]
    covariance = (
        matrix_weights
        * (covariance_left + delta_left[..., :, None] * delta_left[..., None, :])
        + (1.0 - matrix_weights)
        * (covariance_right + delta_right[..., :, None] * delta_right[..., None, :])
    ).mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8 * length_scale * length_scale)
    negative = np.linalg.det(eigenvectors) < 0
    eigenvectors[negative, :, -1] *= -1
    quaternion_xyzw = Rotation.from_matrix(eigenvectors).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    quaternion_wxyz[quaternion_wxyz[:, 0] < 0] *= -1
    return np.sqrt(eigenvalues), quaternion_wxyz


@torch.no_grad()
def apply_merges(reference: Any, plan: PairPlan, k0: dict[str, Any], max_pairs: int | None = None):
    transformed = reference.gather(plan.static_indices, plan.dynamic_indices)
    times = list(k0["specification"]["times"])
    length_scale = float(plan.diagnostics["length_scale"])
    groups = (
        (
            "static",
            plan.static_indices,
            plan.static_pairs,
            plan.static_weights,
            0,
        ),
        (
            "dynamic",
            plan.dynamic_indices,
            plan.dynamic_pairs,
            plan.dynamic_weights,
            reference.static_count,
        ),
    )
    applied = {"static": 0, "dynamic": 0}
    for kind, selected, pairs, weights, global_offset in groups:
        selected_set = set(selected.tolist())
        retained = [index for index, pair in enumerate(pairs) if int(pair[0]) in selected_set]
        if max_pairs is not None:
            retained = retained[:max_pairs]
        device = transformed.model._xyz.device
        dtype = transformed.model._xyz.dtype
        for batch_start in range(0, len(retained), 5_000):
            batch_indices = np.asarray(retained[batch_start : batch_start + 5_000], dtype=np.int64)
            batch_pairs = pairs[batch_indices]
            parents = batch_pairs[:, 0].astype(np.int64)
            left = batch_pairs[:, 2].astype(np.int64)
            right = batch_pairs[:, 3].astype(np.int64)
            rows = np.searchsorted(selected, parents)
            left_weights = weights[batch_indices]
            right_weights = 1.0 - left_weights
            positions_left = k0["positions"][:, global_offset + left].double().numpy()
            positions_right = k0["positions"][:, global_offset + right].double().numpy()
            rotations_left = k0["rotations"][:, global_offset + left].double().numpy()
            rotations_right = k0["rotations"][:, global_offset + right].double().numpy()
            scales_left = k0["scaling"][global_offset + left].double().numpy()
            scales_right = k0["scaling"][global_offset + right].double().numpy()
            merged_scale, merged_rotation = _moment_shape_batch(
                positions_left,
                positions_right,
                scales_left,
                scales_right,
                rotations_left,
                rotations_right,
                left_weights,
                length_scale,
            )
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
            left_t = torch.as_tensor(left, dtype=torch.long, device=device)
            right_t = torch.as_tensor(right, dtype=torch.long, device=device)
            left_weight_t = torch.as_tensor(left_weights, dtype=dtype, device=device)
            right_weight_t = torch.as_tensor(right_weights, dtype=dtype, device=device)
            if kind == "static":
                transformed.model._xyz.data[rows_t] = (
                    left_weight_t[:, None] * reference.model._xyz.data[left_t]
                    + right_weight_t[:, None] * reference.model._xyz.data[right_t]
                )
                transformed.model._xyz_disp.data[rows_t] = (
                    left_weight_t[:, None] * reference.model._xyz_disp.data[left_t]
                    + right_weight_t[:, None] * reference.model._xyz_disp.data[right_t]
                )
                transformed.model._features_dc.data[rows_t] = (
                    left_weight_t[:, None, None] * reference.model._features_dc.data[left_t]
                    + right_weight_t[:, None, None] * reference.model._features_dc.data[right_t]
                )
                transformed.model._features_rest.data[rows_t] = (
                    left_weight_t[:, None, None] * reference.model._features_rest.data[left_t]
                    + right_weight_t[:, None, None] * reference.model._features_rest.data[right_t]
                )
                opacity_left = torch.sigmoid(reference.model._opacity.data[left_t])
                opacity_right = torch.sigmoid(reference.model._opacity.data[right_t])
                opacity = (1.0 - (1.0 - opacity_left) * (1.0 - opacity_right)).clamp(1e-6, 1 - 1e-6)
                transformed.model._opacity.data[rows_t] = torch.logit(opacity)
                transformed.model._scaling.data[rows_t] = torch.log(
                    torch.as_tensor(merged_scale, dtype=dtype, device=device)
                )
                transformed.model._rotation.data[rows_t] = torch.as_tensor(
                    merged_rotation, dtype=dtype, device=device
                )
            else:
                transformed.model._xyz_motion.data[rows_t] = (
                    left_weight_t[:, None, None] * reference.model._xyz_motion.data[left_t]
                    + right_weight_t[:, None, None] * reference.model._xyz_motion.data[right_t]
                )
                transformed.model._features_dc_motion.data[rows_t] = (
                    left_weight_t[:, None, None] * reference.model._features_dc_motion.data[left_t]
                    + right_weight_t[:, None, None] * reference.model._features_dc_motion.data[right_t]
                )
                transformed.model._features_rest_motion.data[rows_t] = (
                    left_weight_t[:, None, None] * reference.model._features_rest_motion.data[left_t]
                    + right_weight_t[:, None, None] * reference.model._features_rest_motion.data[right_t]
                )
                opacity_left = torch.sigmoid(reference.model._opacity_motion.data[left_t])
                opacity_right = torch.sigmoid(reference.model._opacity_motion.data[right_t])
                opacity = (1.0 - (1.0 - opacity_left) * (1.0 - opacity_right)).clamp(1e-6, 1 - 1e-6)
                transformed.model._opacity_motion.data[rows_t] = torch.logit(opacity)
                transformed.model._scaling_motion.data[rows_t] = torch.log(
                    torch.as_tensor(merged_scale, dtype=dtype, device=device)
                )
                quaternion = torch.as_tensor(merged_rotation, dtype=dtype, device=device)
                transformed.model._rotation_motion.data[rows_t] = quaternion[:, None, :].repeat(
                    1, transformed.model._rotation_motion.shape[1], 1
                )
            applied[kind] += int(batch_indices.size)
    return transformed, applied


def moment_toy() -> dict[str, Any]:
    positions = np.zeros((2, 3), dtype=np.float64)
    scales = np.asarray([0.2, 0.3, 0.4], dtype=np.float64)
    rotations = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1))
    mean, merged_scale, quaternion = _moment_shape(
        positions, positions, scales, scales, rotations, rotations, 0.5, 1.0
    )
    matrix = _rotation_matrices(quaternion[None])[0]
    covariance = matrix @ np.diag(merged_scale**2) @ matrix.T
    expected = np.diag(scales**2)
    alpha_left, alpha_right = 0.6, 0.5
    color_left, color_right = 0.2, 0.8
    composite = color_left * alpha_left + (1 - alpha_left) * color_right * alpha_right
    merged_alpha = 1 - (1 - alpha_left) * (1 - alpha_right)
    merged_color = 0.5 * color_left + 0.5 * color_right
    merged_ray = merged_color * merged_alpha
    result = {
        "identical_mean_max_abs": float(np.abs(mean).max()),
        "identical_covariance_max_abs": float(np.abs(covariance - expected).max()),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
        "rotation_orthogonality_max_abs": float(np.abs(matrix.T @ matrix - np.eye(3)).max()),
        "quaternion_norm_error": abs(float(np.linalg.norm(quaternion)) - 1.0),
        "ray_color_nonconservation_error": abs(composite - merged_ray),
    }
    result["passed"] = bool(
        result["identical_mean_max_abs"] <= 1e-12
        and result["identical_covariance_max_abs"] <= 1e-12
        and result["minimum_eigenvalue"] > 0
        and result["rotation_orthogonality_max_abs"] <= 1e-12
        and result["quaternion_norm_error"] <= 1e-12
        and result["ray_color_nonconservation_error"] > 0
    )
    return result
