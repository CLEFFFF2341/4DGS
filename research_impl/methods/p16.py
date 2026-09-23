from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scene import getmodel

from ..adapter import DYNAMIC_PARAMETERS, STATIC_PARAMETERS, ModelAdapter


def _haar_split(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    original_keyframes = values.shape[1]
    padded = values if original_keyframes % 2 == 0 else torch.cat((values, values[:, -1:]), dim=1)
    approximation = (padded[:, 0::2] + padded[:, 1::2]) / math.sqrt(2.0)
    detail = (padded[:, 0::2] - padded[:, 1::2]) / math.sqrt(2.0)
    return approximation, detail, original_keyframes


def _haar_decode(approximation: torch.Tensor, detail: torch.Tensor | None, original_keyframes: int) -> torch.Tensor:
    if detail is None:
        even = odd = approximation / math.sqrt(2.0)
    else:
        even = (approximation + detail) / math.sqrt(2.0)
        odd = (approximation - detail) / math.sqrt(2.0)
    return torch.stack((even, odd), dim=2).flatten(1, 2)[:, :original_keyframes]


def _fp16_decode(values: torch.Tensor) -> torch.Tensor:
    half = values.half()
    if not torch.isfinite(half).all():
        raise RuntimeError("FP16 position encoding overflowed")
    return half.float()


@torch.no_grad()
def save_compact_bundle(adapter, encoded: dict, rule: str, path: Path) -> int:
    parameters = {}
    for name in STATIC_PARAMETERS + DYNAMIC_PARAMETERS:
        if name == "_xyz_motion":
            continue
        value = getattr(adapter.model, name)
        parameters[name] = {"tensor": value.detach().cpu(), "requires_grad": bool(value.requires_grad)}
    payload = {
        "schema": "p16-compact-bundle-v1",
        "rule": rule,
        "encoded_motion": encoded,
        "parameters": parameters,
        "active_sh_degree": int(adapter.model.active_sh_degree),
        "spatial_lr_scale": float(adapter.model.spatial_lr_scale),
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "static_rows": torch.from_numpy(adapter.static_rows.copy()),
        "dynamic_rows": torch.from_numpy(adapter.dynamic_rows.copy()),
    }
    torch.save(payload, path)
    return path.stat().st_size


@torch.no_grad()
def load_compact_bundle(path: Path, args) -> ModelAdapter:
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "p16-compact-bundle-v1":
        raise RuntimeError(f"Unsupported P16 compact bundle: {path}")
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
    encoded = payload["encoded_motion"]
    if payload["rule"] == "R0":
        decoded = _haar_decode(encoded["approximation"].cuda(), None, int(encoded["original_keyframes"]))
    elif payload["rule"] == "R1":
        decoded = encoded["position_fp16"].cuda().float()
    else:
        raise KeyError(payload["rule"])
    model._xyz_motion = nn.Parameter(decoded)
    return ModelAdapter(
        model=model,
        checkpoint_sha256=payload["checkpoint_sha256"],
        static_rows=np.asarray(payload["static_rows"].numpy(), dtype=np.int64),
        dynamic_rows=np.asarray(payload["dynamic_rows"].numpy(), dtype=np.int64),
        args=args,
    )


@torch.no_grad()
def encode_decode(adapter, rule: str):
    transformed = adapter.gather(range(adapter.static_count), range(adapter.dynamic_count))
    original = adapter.model._xyz_motion.detach()
    if rule == "R0":
        approximation, _, original_keyframes = _haar_split(original)
        decoded = _haar_decode(approximation, None, original_keyframes)
        encoded = {"approximation": approximation.cpu(), "original_keyframes": original_keyframes}
        encoded_bytes = approximation.numel() * approximation.element_size() + 8
    elif rule == "R1":
        decoded = _fp16_decode(original)
        half = original.half()
        encoded = {"position_fp16": half.cpu(), "original_keyframes": original.shape[1]}
        encoded_bytes = half.numel() * half.element_size() + 8
    else:
        raise KeyError(rule)
    transformed.model._xyz_motion = nn.Parameter(decoded.detach().clone().requires_grad_(original.requires_grad))
    return transformed, encoded, {"encoded_position_bytes": int(encoded_bytes), "decoded_position_bytes": int(decoded.numel() * decoded.element_size())}


@torch.no_grad()
def correctness_checks(adapter):
    original = adapter.model._xyz_motion.detach()
    approximation, detail, original_keyframes = _haar_split(original)
    restored = _haar_decode(approximation, detail, original_keyframes)
    error = (restored - original).abs()

    constant = torch.full((2, 5, 3), 3.25, device=original.device)
    constant_a, constant_d, constant_j = _haar_split(constant)
    constant_decoded = _haar_decode(constant_a, None, constant_j)

    linear = torch.arange(5, dtype=original.dtype, device=original.device).view(1, 5, 1).repeat(1, 1, 3)
    linear_a, _, linear_j = _haar_split(linear)
    linear_decoded = _haar_decode(linear_a, None, linear_j)

    odd = torch.randn((2, 7, 3), generator=torch.Generator(device=original.device).manual_seed(1607), device=original.device)
    odd_a, odd_d, odd_j = _haar_split(odd)
    odd_decoded = _haar_decode(odd_a, odd_d, odd_j)

    overflow_rejected = False
    try:
        _fp16_decode(torch.tensor([[[70000.0, 0.0, 0.0]]], device=original.device))
    except RuntimeError:
        overflow_rejected = True

    seed_outputs = []
    seed_input = torch.arange(18, dtype=original.dtype, device=original.device).reshape(2, 3, 3)
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        seed_a, _, seed_j = _haar_split(seed_input)
        seed_outputs.append(_haar_decode(seed_a, None, seed_j))

    checks = {
        "full_haar_max_abs": float(error.max()),
        "full_haar_mean_abs": float(error.mean()),
        "constant_detail_max_abs": float(constant_d.abs().max()),
        "constant_compressed_max_abs": float((constant_decoded - constant).abs().max()),
        "linear_pair_averaging_max_abs": float((linear_decoded - linear).abs().max()),
        "linear_pair_averaging_is_nonexact": bool(not torch.equal(linear_decoded, linear)),
        "odd_keyframes_input": int(odd_j),
        "odd_approximation_keyframes": int(odd_a.shape[1]),
        "odd_roundtrip_max_abs": float((odd_decoded - odd).abs().max()),
        "fp16_overflow_rejected": overflow_rejected,
        "seed_invariant_0_1_2": bool(torch.equal(seed_outputs[0], seed_outputs[1]) and torch.equal(seed_outputs[0], seed_outputs[2])),
        "position_semantics": "_xyz_motion is decoded as absolute dynamic positions consumed directly by get_dynamic_xyz_at_t; no static position or displacement is added",
    }
    checks["passed"] = bool(
        checks["full_haar_max_abs"] <= 1e-5
        and checks["constant_detail_max_abs"] <= 1e-7
        and checks["constant_compressed_max_abs"] <= 1e-6
        and checks["linear_pair_averaging_is_nonexact"]
        and checks["odd_approximation_keyframes"] == 4
        and checks["odd_roundtrip_max_abs"] <= 1e-5
        and checks["fp16_overflow_rejected"]
        and checks["seed_invariant_0_1_2"]
    )
    return checks
