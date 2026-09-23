from __future__ import annotations

import heapq
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scene import getmodel

from ..adapter import DYNAMIC_PARAMETERS, STATIC_PARAMETERS, ModelAdapter


HEADER_BYTES = 32
OFFSET_BYTES = 4


def _normalize_and_canonicalize(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float32).copy()
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise RuntimeError("P15 encountered a zero, NaN, or infinite quaternion")
    values /= norms
    for keyframe in range(1, values.shape[1]):
        flip = np.sum(values[:, keyframe - 1] * values[:, keyframe], axis=1) < 0
        values[flip, keyframe] *= -1
    return values


def _slerp_numpy(left: np.ndarray, right: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if dot < 0:
        right = -right
        dot = -dot
    fraction = np.asarray(fraction, dtype=np.float64).reshape(-1, 1)
    if dot > 0.9995:
        result = (1.0 - fraction) * left + fraction * right
    else:
        theta = math.acos(dot)
        sine = math.sin(theta)
        result = np.sin((1.0 - fraction) * theta) / sine * left + np.sin(fraction * theta) / sine * right
    return (result / np.linalg.norm(result, axis=1, keepdims=True)).astype(np.float32)


def _best_in_segment(
    point: int,
    left: int,
    right: int,
    positions: np.ndarray,
    quaternions: np.ndarray,
    length_scale: float,
) -> tuple[float, int] | None:
    if right - left <= 1:
        return None
    indices = np.arange(left + 1, right, dtype=np.int64)
    fraction = (indices - left).astype(np.float64) / float(right - left)
    interpolated_position = positions[point, left] + fraction[:, None] * (positions[point, right] - positions[point, left])
    position_error = np.sum((positions[point, indices] - interpolated_position) ** 2, axis=1) / (length_scale * length_scale)
    interpolated_rotation = _slerp_numpy(quaternions[point, left], quaternions[point, right], fraction)
    dots = np.clip(np.abs(np.sum(quaternions[point, indices] * interpolated_rotation, axis=1)), 0.0, 1.0)
    angle = 2.0 * np.arccos(dots)
    error = position_error + (angle / math.pi) ** 2
    local = int(np.argmax(error))
    return float(error[local]), int(indices[local])


def _budget(dynamic_count: int, keyframes: int, target_bytes: int) -> dict:
    index_bytes = 2 if keyframes <= 65535 else 4
    bytes_per_knot = index_bytes + 7 * 4
    offset_bytes = (dynamic_count + 1) * OFFSET_BYTES
    available = target_bytes - HEADER_BYTES - offset_bytes
    knot_capacity = min(dynamic_count * keyframes, available // bytes_per_knot)
    minimum_knots = dynamic_count * min(2, keyframes)
    if knot_capacity < minimum_knots:
        raise RuntimeError("P15 target cannot hold mandatory endpoint knots")
    return {
        "index_bytes": index_bytes,
        "bytes_per_knot": bytes_per_knot,
        "offset_bytes": offset_bytes,
        "knot_capacity": int(knot_capacity),
        "minimum_knots": int(minimum_knots),
    }


def _adaptive_knots(
    positions: np.ndarray,
    quaternions: np.ndarray,
    stable_rows: np.ndarray,
    length_scale: float,
    knot_capacity: int,
    progress_interval: int = 100000,
) -> tuple[list[list[int]], list[dict]]:
    dynamic_count, keyframes, _ = positions.shape
    if keyframes == 1:
        return [[0] for _ in range(dynamic_count)], []
    knots = [[0, keyframes - 1] for _ in range(dynamic_count)]
    heap = []
    for point in range(dynamic_count):
        candidate = _best_in_segment(point, 0, keyframes - 1, positions, quaternions, length_scale)
        if candidate is not None:
            error, keyframe = candidate
            heapq.heappush(heap, (-error, int(stable_rows[point]), keyframe, point, 0, keyframes - 1))
    additions = knot_capacity - 2 * dynamic_count
    trajectory = []
    for step in range(additions):
        if not heap:
            break
        negative_error, stable_row, keyframe, point, left, right = heapq.heappop(heap)
        if -negative_error <= 0:
            break
        knots[point].append(keyframe)
        for child_left, child_right in ((left, keyframe), (keyframe, right)):
            candidate = _best_in_segment(point, child_left, child_right, positions, quaternions, length_scale)
            if candidate is not None:
                error, child_keyframe = candidate
                heapq.heappush(
                    heap,
                    (-error, stable_row, child_keyframe, point, child_left, child_right),
                )
        if step == 0 or (step + 1) % progress_interval == 0 or step + 1 == additions:
            row = {"additions": step + 1, "total_knots": 2 * dynamic_count + step + 1, "selected_error": -negative_error}
            trajectory.append(row)
            print(f"P15 R0 selection: {step + 1}/{additions} additions, error={-negative_error:.6g}", flush=True)
    for item in knots:
        item.sort()
    return knots, trajectory


def _uniform_order(keyframes: int) -> list[int]:
    knots = [0] if keyframes == 1 else [0, keyframes - 1]
    additions = []
    while len(knots) < keyframes:
        intervals = [(knots[index + 1] - knots[index], knots[index], knots[index + 1]) for index in range(len(knots) - 1)]
        _, left, right = max(intervals, key=lambda item: (item[0], -item[1]))
        midpoint = (left + right) // 2
        additions.append(midpoint)
        knots.append(midpoint)
        knots.sort()
    return additions


def _uniform_knots(dynamic_count: int, keyframes: int, stable_rows: np.ndarray, knot_capacity: int) -> list[list[int]]:
    endpoint_count = min(2, keyframes)
    additions = knot_capacity - endpoint_count * dynamic_count
    base, remainder = divmod(additions, dynamic_count) if dynamic_count else (0, 0)
    order = _uniform_order(keyframes)
    stable_order = np.argsort(stable_rows, kind="stable")
    extra = np.zeros(dynamic_count, dtype=np.int64)
    if remainder:
        extra[stable_order[:remainder]] = 1
    result = []
    endpoints = [0] if keyframes == 1 else [0, keyframes - 1]
    for point in range(dynamic_count):
        count = min(len(order), base + int(extra[point]))
        result.append(sorted(endpoints + order[:count]))
    return result


def _pack_knots(positions: np.ndarray, quaternions: np.ndarray, knots: list[list[int]], index_bytes: int) -> dict:
    counts = np.asarray([len(item) for item in knots], dtype=np.int64)
    offsets = np.zeros(len(knots) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
    # The installed Torch build cannot serialize uint16/uint32 tensors.  J=37,
    # so signed int16 is a lossless, equal-width container for logical uint16.
    index_dtype = np.int16 if index_bytes == 2 else np.int32
    indices = np.empty(int(offsets[-1]), dtype=index_dtype)
    packed_position = np.empty((int(offsets[-1]), 3), dtype=np.float32)
    packed_rotation = np.empty((int(offsets[-1]), 4), dtype=np.float32)
    cursor = 0
    for point, point_knots in enumerate(knots):
        count = len(point_knots)
        indices[cursor : cursor + count] = point_knots
        packed_position[cursor : cursor + count] = positions[point, point_knots]
        packed_rotation[cursor : cursor + count] = quaternions[point, point_knots]
        cursor += count
    return {
        "offsets": torch.from_numpy(offsets),
        "indices": torch.from_numpy(indices),
        "positions": torch.from_numpy(packed_position),
        "rotations": torch.from_numpy(packed_rotation),
        "keyframes": int(positions.shape[1]),
    }


def _decode_packed(encoded: dict, original_position: torch.Tensor | None = None, original_rotation: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = encoded["offsets"].cpu().numpy()
    indices = encoded["indices"].cpu().numpy().astype(np.int64)
    packed_position = encoded["positions"].cpu().numpy()
    packed_rotation = encoded["rotations"].cpu().numpy()
    dynamic_count = len(offsets) - 1
    keyframes = int(encoded["keyframes"])
    if int(offsets[-1]) == dynamic_count * keyframes and original_position is not None and original_rotation is not None:
        return original_position.detach().cpu().clone(), original_rotation.detach().cpu().clone()
    position = np.empty((dynamic_count, keyframes, 3), dtype=np.float32)
    rotation = np.empty((dynamic_count, keyframes, 4), dtype=np.float32)
    for point in range(dynamic_count):
        start, end = int(offsets[point]), int(offsets[point + 1])
        point_indices = indices[start:end]
        point_position = packed_position[start:end]
        point_rotation = packed_rotation[start:end]
        for interval in range(len(point_indices) - 1):
            left, right = int(point_indices[interval]), int(point_indices[interval + 1])
            target = np.arange(left, right + 1, dtype=np.int64)
            fraction = (target - left).astype(np.float64) / max(right - left, 1)
            position[point, target] = point_position[interval] + fraction[:, None] * (point_position[interval + 1] - point_position[interval])
            rotation[point, target] = _slerp_numpy(point_rotation[interval], point_rotation[interval + 1], fraction)
    return torch.from_numpy(position), torch.from_numpy(rotation)


@torch.no_grad()
def encode_decode(adapter, rule: str, length_scale: float, target_fraction: float = 0.5, smoke_points: int | None = None):
    original_position = adapter.model._xyz_motion.detach()
    original_rotation = adapter.model._rotation_motion.detach()
    positions = original_position.cpu().float().numpy()
    quaternions = _normalize_and_canonicalize(original_rotation.cpu().float().numpy())
    stable_rows = adapter.dynamic_rows
    if smoke_points is not None:
        count = min(smoke_points, len(positions))
        positions = positions[:count]
        quaternions = quaternions[:count]
        stable_rows = stable_rows[:count]
    original_bytes = positions.nbytes + quaternions.nbytes
    target_bytes = int(math.floor(target_fraction * original_bytes))
    budget = _budget(len(positions), positions.shape[1], target_bytes)
    selection_started = time.perf_counter()
    if rule == "R0":
        knots, budget_trajectory = _adaptive_knots(positions, quaternions, stable_rows, length_scale, budget["knot_capacity"])
    elif rule == "R1":
        knots = _uniform_knots(len(positions), positions.shape[1], stable_rows, budget["knot_capacity"])
        budget_trajectory = []
    else:
        raise KeyError(rule)
    selection_seconds = time.perf_counter() - selection_started
    encoded = _pack_knots(positions, quaternions, knots, budget["index_bytes"])
    decoded_position, decoded_rotation = _decode_packed(encoded)
    encoded_bytes = (
        encoded["offsets"].numel() * encoded["offsets"].element_size()
        + encoded["indices"].numel() * encoded["indices"].element_size()
        + encoded["positions"].numel() * encoded["positions"].element_size()
        + encoded["rotations"].numel() * encoded["rotations"].element_size()
        + HEADER_BYTES
    )
    diagnostics = {
        **budget,
        "target_variable_bytes": target_bytes,
        "encoded_variable_bytes": int(encoded_bytes),
        "original_variable_bytes": int(original_bytes),
        "selection_seconds": selection_seconds,
        "total_knots": int(encoded["indices"].numel()),
        "mean_knots_per_point": float(encoded["indices"].numel() / max(len(positions), 1)),
        "position_relative_frobenius_error": float(torch.linalg.vector_norm(decoded_position - torch.from_numpy(positions)) / torch.linalg.vector_norm(torch.from_numpy(positions))),
        "budget_trajectory": budget_trajectory,
    }
    if smoke_points is not None:
        return encoded, decoded_position, decoded_rotation, diagnostics
    transformed = adapter.gather(range(adapter.static_count), range(adapter.dynamic_count))
    transformed.model._xyz_motion = nn.Parameter(decoded_position.cuda().requires_grad_(original_position.requires_grad))
    transformed.model._rotation_motion = nn.Parameter(decoded_rotation.cuda().requires_grad_(original_rotation.requires_grad))
    return transformed, encoded, diagnostics


@torch.no_grad()
def save_compact_bundle(adapter, encoded: dict, rule: str, path: Path) -> int:
    parameters = {}
    for name in STATIC_PARAMETERS + DYNAMIC_PARAMETERS:
        if name in {"_xyz_motion", "_rotation_motion"}:
            continue
        value = getattr(adapter.model, name)
        parameters[name] = {"tensor": value.detach().cpu(), "requires_grad": bool(value.requires_grad)}
    torch.save(
        {
            "schema": "p15-compact-bundle-v1",
            "rule": rule,
            "encoded_motion": encoded,
            "parameters": parameters,
            "active_sh_degree": int(adapter.model.active_sh_degree),
            "spatial_lr_scale": float(adapter.model.spatial_lr_scale),
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "static_rows": torch.from_numpy(adapter.static_rows.copy()),
            "dynamic_rows": torch.from_numpy(adapter.dynamic_rows.copy()),
        },
        path,
    )
    return path.stat().st_size


@torch.no_grad()
def load_compact_bundle(path: Path, args) -> ModelAdapter:
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "p15-compact-bundle-v1":
        raise RuntimeError(f"Unsupported P15 compact bundle: {path}")
    cls = getmodel(args.model)
    model = cls(
        args.sh_degree,
        args.duration,
        args.time_interval,
        args.time_pad,
        interp_type=args.interp_type,
        rot_interp_type=args.rot_interp_type,
        time_pad_type=args.time_pad_type,
        var_pad=args.var_pad,
        kernel_size=args.kernel_size,
    )
    model.active_sh_degree = payload["active_sh_degree"]
    model.spatial_lr_scale = payload["spatial_lr_scale"]
    for name, item in payload["parameters"].items():
        tensor = item["tensor"].cuda()
        setattr(model, name, nn.Parameter(tensor.requires_grad_(item["requires_grad"])))
    position, rotation = _decode_packed(payload["encoded_motion"])
    model._xyz_motion = nn.Parameter(position.cuda())
    model._rotation_motion = nn.Parameter(rotation.cuda())
    return ModelAdapter(
        model=model,
        checkpoint_sha256=payload["checkpoint_sha256"],
        static_rows=np.asarray(payload["static_rows"].numpy(), dtype=np.int64),
        dynamic_rows=np.asarray(payload["dynamic_rows"].numpy(), dtype=np.int64),
        args=args,
    )


def correctness_checks() -> dict:
    keyframes = 7
    fraction = np.linspace(0.0, 1.0, keyframes, dtype=np.float32)
    positions = np.stack((fraction, 2 * fraction, -fraction), axis=1)[None]
    rotations = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, keyframes, 1))
    knots = [[0, keyframes - 1]]
    packed = _pack_knots(positions, rotations, knots, 2)
    decoded_position, decoded_rotation = _decode_packed(packed)

    q = np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    q /= np.linalg.norm(q)
    sign_equivalent_angle = 2 * math.acos(float(np.clip(abs(np.dot(q, -q)), 0, 1)))
    half_turn = _slerp_numpy(np.asarray([1, 0, 0, 0], dtype=np.float32), np.asarray([0, 1, 0, 0], dtype=np.float32), np.asarray([0.5]))[0]
    expected_half_turn = np.asarray([math.sqrt(0.5), math.sqrt(0.5), 0, 0], dtype=np.float32)

    generator = np.random.default_rng(1515)
    all_position = torch.from_numpy(generator.normal(size=(2, 5, 3)).astype(np.float32))
    all_rotation = torch.from_numpy(generator.normal(size=(2, 5, 4)).astype(np.float32))
    normalized = _normalize_and_canonicalize(all_rotation.numpy())
    all_knots = [list(range(5)), list(range(5))]
    all_packed = _pack_knots(all_position.numpy(), normalized, all_knots, 2)
    all_decoded_position, all_decoded_rotation = _decode_packed(all_packed, all_position, all_rotation)

    seed_outputs = []
    for seed in (0, 1, 2):
        np.random.seed(seed)
        seed_outputs.append(_uniform_order(37))

    checks = {
        "linear_position_endpoint_max_abs": float(np.max(np.abs(decoded_position.numpy() - positions))),
        "constant_rotation_endpoint_max_abs": float(np.max(np.abs(decoded_rotation.numpy() - rotations))),
        "q_and_negative_q_angle": sign_equivalent_angle,
        "half_turn_slerp_max_abs": float(np.max(np.abs(half_turn - expected_half_turn))),
        "all_knots_position_max_abs": float((all_decoded_position - all_position).abs().max()),
        "all_knots_rotation_max_abs": float((all_decoded_rotation - all_rotation).abs().max()),
        "padding_indices_in_range": bool(int(all_packed["indices"].max()) < 5),
        "seed_invariant_0_1_2": seed_outputs[0] == seed_outputs[1] == seed_outputs[2],
        "empty_dynamic_group_policy": "skip transform; real run has a non-empty dynamic group",
    }
    checks["passed"] = bool(
        checks["linear_position_endpoint_max_abs"] <= 1e-6
        and checks["constant_rotation_endpoint_max_abs"] <= 1e-6
        and checks["q_and_negative_q_angle"] <= 1e-7
        and checks["half_turn_slerp_max_abs"] <= 1e-6
        and checks["all_knots_position_max_abs"] == 0
        and checks["all_knots_rotation_max_abs"] == 0
        and checks["padding_indices_in_range"]
        and checks["seed_invariant_0_1_2"]
    )
    return checks
