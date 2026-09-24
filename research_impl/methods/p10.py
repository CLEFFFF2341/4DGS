from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.sh_utils import SH2RGB


@dataclass(frozen=True)
class SelectionResult:
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    demands: list[dict]
    diagnostics: dict


def active_episodes(values: np.ndarray) -> list[np.ndarray]:
    """Return active-index episodes; at least two missing samples split episodes."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("visibility values must be a finite nonnegative vector")
    maximum = float(values.max(initial=0.0))
    if maximum <= 0:
        return []
    active = np.flatnonzero((values >= 0.05 * maximum) & (values > 0))
    if active.size == 0:
        return []
    split = np.flatnonzero(np.diff(active) >= 3) + 1
    return [part.astype(np.int64) for part in np.split(active, split) if part.size]


def enumerate_reappearing_demands(s_it: np.ndarray, e_it: np.ndarray, static_count: int) -> list[dict]:
    if s_it.shape != e_it.shape:
        raise ValueError(f"K1/K2 shape mismatch: {s_it.shape} vs {e_it.shape}")
    if s_it.shape[0] != 24:
        raise ValueError("P10 requires exactly T24")
    demands = []
    nonzero = np.flatnonzero(s_it.max(axis=0) > 0)
    for point in nonzero:
        episodes = active_episodes(s_it[:, point])
        if len(episodes) < 2:
            continue
        kind = "static" if point < static_count else "dynamic"
        local_row = int(point if kind == "static" else point - static_count)
        for episode_index, episode in enumerate(episodes):
            demands.append(
                {
                    "global_index": int(point),
                    "kind": kind,
                    "local_row": local_row,
                    "episode_index": episode_index,
                    "episode_samples": episode.tolist(),
                    "episode_score": float(e_it[episode, point].sum()),
                    "reappearance_sample": int(episodes[1][0]),
                }
            )
    demands.sort(key=lambda row: (-row["episode_score"], row["kind"], row["local_row"], row["episode_index"]))
    return demands


def project_centers(points: torch.Tensor, full_proj_transform: torch.Tensor, width: int, height: int) -> torch.Tensor:
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    clip = homogeneous @ full_proj_transform
    reciprocal = 1.0 / (clip[:, 3:4] + 1e-7)
    ndc = clip[:, :2] * reciprocal
    x = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
    y = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5
    return torch.stack((x, y), dim=1)


@torch.no_grad()
def attach_feasible_candidates(adapter, scene, demands: list[dict], s_it: np.ndarray, k4: dict, camera_names: list[str], times: list[int], resolution: list[int], select_camera) -> dict:
    static_neighbors = k4["static_neighbors"].long().numpy()
    dynamic_neighbors = k4["dynamic_neighbors"].long().numpy()
    all_candidate_indices = set()
    raw_candidates = []
    for demand in demands:
        if demand["kind"] == "static":
            candidates = static_neighbors[demand["local_row"]].astype(np.int64)
        else:
            candidates = adapter.static_count + dynamic_neighbors[demand["local_row"]].astype(np.int64)
        candidates = np.unique(np.append(candidates, demand["global_index"]))
        raw_candidates.append(candidates)
        all_candidate_indices.update(candidates.tolist())
    unique = np.asarray(sorted(all_candidate_indices), dtype=np.int64)
    lookup = {int(point): index for index, point in enumerate(unique)}
    device_indices = torch.from_numpy(unique).long().cuda()
    positions = []
    projections = []
    width, height = map(int, resolution)
    for timestamp in times:
        xyz = adapter.model.get_xyz_at_t(timestamp, training=False).index_select(0, device_indices)
        positions.append(xyz.detach().cpu())
        camera_projection = []
        for name in camera_names:
            camera = select_camera(scene, name, timestamp)
            camera_projection.append(project_centers(xyz, camera.full_proj_transform, width, height).cpu())
        projections.append(torch.stack(camera_projection))
    positions_tensor = torch.stack(positions)
    projections_tensor = torch.stack(projections).numpy()
    dc = SH2RGB(adapter.model.get_features()[:, 0, :].detach()).cpu().numpy()
    feasible = 0
    for demand, candidates in zip(demands, raw_candidates):
        point = demand["global_index"]
        point_slot = lookup[point]
        episode = np.asarray(demand["episode_samples"], dtype=np.int64)
        accepted = []
        rejection = {"visibility": 0, "projection": 0, "color": 0}
        for candidate in candidates:
            candidate = int(candidate)
            if not bool(np.all(s_it[episode, candidate] > 0)):
                rejection["visibility"] += 1
                continue
            candidate_slot = lookup[candidate]
            difference = projections_tensor[episode, :, candidate_slot] - projections_tensor[episode, :, point_slot]
            per_time_minimum = np.linalg.norm(difference, axis=2).min(axis=1)
            if not bool(np.all(per_time_minimum <= 8.0)):
                rejection["projection"] += 1
                continue
            if float(np.linalg.norm(dc[candidate] - dc[point])) > 0.1:
                rejection["color"] += 1
                continue
            accepted.append(candidate)
        demand["candidate_global_indices"] = accepted
        demand["candidate_rejections"] = rejection
        demand["feasible"] = bool(accepted)
        feasible += int(bool(accepted))
    return {
        "unique_candidate_points": int(unique.size),
        "position_shape": list(positions_tensor.shape),
        "feasible_demands": feasible,
        "projected_camera_count": len(camera_names),
        "statistics_resolution": [width, height],
    }


def repair_selection(
    reference,
    base_static_rows: np.ndarray,
    base_dynamic_rows: np.ndarray,
    demands: list[dict],
    scores: np.ndarray,
) -> SelectionResult:
    kept = set(int(row) for row in base_static_rows)
    kept.update(reference.static_count + int(row) for row in base_dynamic_rows)
    static_order = np.lexsort((reference.static_rows, scores[: reference.static_count]))
    dynamic_order = reference.static_count + np.lexsort((reference.dynamic_rows, scores[reference.static_count :]))
    removal_order = {"static": static_order.astype(np.int64), "dynamic": dynamic_order.astype(np.int64)}
    processed_candidates: list[set[int]] = []
    reverse: dict[int, set[int]] = {}
    satisfaction_counts: list[int] = []
    repairs = []
    already_satisfied = 0
    infeasible = 0
    for demand_index, demand in enumerate(demands):
        candidates = set(int(value) for value in demand["candidate_global_indices"])
        if not candidates:
            demand["repair_status"] = "infeasible_no_candidate"
            infeasible += 1
            continue
        present = candidates.intersection(kept)
        if present:
            demand["repair_status"] = "already_satisfied"
            already_satisfied += 1
        else:
            insert = sorted(candidates, key=lambda value: (-scores[value], value))[0]
            kept.add(insert)
            for prior_index in reverse.get(insert, set()):
                satisfaction_counts[prior_index] += 1
            remove = None
            for candidate_remove in removal_order[demand["kind"]]:
                candidate_remove = int(candidate_remove)
                if candidate_remove == insert or candidate_remove not in kept:
                    continue
                unsafe = any(satisfaction_counts[prior_index] <= 1 for prior_index in reverse.get(candidate_remove, set()))
                if not unsafe:
                    remove = candidate_remove
                    break
            if remove is None:
                kept.remove(insert)
                for prior_index in reverse.get(insert, set()):
                    satisfaction_counts[prior_index] -= 1
                demand["repair_status"] = "infeasible_no_safe_removal"
                infeasible += 1
                continue
            kept.remove(remove)
            for prior_index in reverse.get(remove, set()):
                satisfaction_counts[prior_index] -= 1
            demand["repair_status"] = "repaired"
            demand["inserted_global_index"] = insert
            demand["removed_global_index"] = remove
            repairs.append({"demand_index": demand_index, "inserted_global_index": insert, "removed_global_index": remove})
        processed_index = len(processed_candidates)
        processed_candidates.append(candidates)
        satisfaction_counts.append(len(candidates.intersection(kept)))
        for candidate in candidates:
            reverse.setdefault(candidate, set()).add(processed_index)
        if satisfaction_counts[-1] <= 0:
            raise RuntimeError("repair failed to satisfy the current demand")
        if any(count <= 0 for count in satisfaction_counts):
            raise RuntimeError("repair broke a previously satisfied demand")
    static = np.asarray(sorted(value for value in kept if value < reference.static_count), dtype=np.int64)
    dynamic = np.asarray(sorted(value - reference.static_count for value in kept if value >= reference.static_count), dtype=np.int64)
    diagnostics = {
        "selected_demands": len(demands),
        "feasible_demands": int(sum(bool(row["candidate_global_indices"]) for row in demands)),
        "already_satisfied": already_satisfied,
        "repairs": len(repairs),
        "infeasible": infeasible,
        "repair_trace": repairs,
        "actual_total": int(static.size + dynamic.size),
        "static_count": int(static.size),
        "dynamic_count": int(dynamic.size),
        "all_processed_constraints_satisfied": bool(all(count > 0 for count in satisfaction_counts)),
    }
    return SelectionResult(static, dynamic, demands, diagnostics)


def correctness_checks() -> dict:
    two = active_episodes(np.asarray([1, 1, 0, 0, 1, 1], dtype=np.float64))
    one = active_episodes(np.asarray([1, 1, 0, 1, 1], dtype=np.float64))
    zero = active_episodes(np.zeros(6, dtype=np.float64))
    points = torch.tensor([[0.0, 0.0, 0.0], [0.5, -0.5, 0.0]])
    identity = torch.eye(4)
    projected = project_centers(points, identity, 100, 80)
    checks = {
        "two_segments_with_two_sample_gap": len(two) == 2,
        "one_missing_sample_does_not_split": len(one) == 1,
        "zero_visibility_generates_no_episode": len(zero) == 0,
        "identity_projection_center": bool(torch.allclose(projected[0], torch.tensor([49.5, 39.5]))),
        "identity_projection_offset": bool(torch.allclose(projected[1], torch.tensor([74.5, 19.5]))),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks
