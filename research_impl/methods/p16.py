from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def encode_decode(adapter, rule: str):
    transformed = adapter.gather(range(adapter.static_count), range(adapter.dynamic_count))
    original = adapter.model._xyz_motion.detach()
    if rule == "R0":
        padded = original if original.shape[1] % 2 == 0 else torch.cat((original, original[:, -1:]), dim=1)
        approximation = (padded[:, 0::2] + padded[:, 1::2]) / math.sqrt(2.0)
        decoded = torch.repeat_interleave(approximation / math.sqrt(2.0), 2, dim=1)[:, : original.shape[1]]
        encoded = {"approximation": approximation.cpu(), "original_keyframes": original.shape[1]}
        encoded_bytes = approximation.numel() * approximation.element_size() + 8
    elif rule == "R1":
        half = original.half()
        if not torch.isfinite(half).all():
            raise RuntimeError("FP16 position encoding overflowed")
        decoded = half.float()
        encoded = {"position_fp16": half.cpu(), "original_keyframes": original.shape[1]}
        encoded_bytes = half.numel() * half.element_size() + 8
    else:
        raise KeyError(rule)
    transformed.model._xyz_motion = nn.Parameter(decoded.detach().clone().requires_grad_(original.requires_grad))
    return transformed, encoded, {"encoded_position_bytes": int(encoded_bytes), "decoded_position_bytes": int(decoded.numel() * decoded.element_size())}


@torch.no_grad()
def correctness_checks(adapter):
    original = adapter.model._xyz_motion.detach()
    padded = original if original.shape[1] % 2 == 0 else torch.cat((original, original[:, -1:]), dim=1)
    approximation = (padded[:, 0::2] + padded[:, 1::2]) / math.sqrt(2.0)
    detail = (padded[:, 0::2] - padded[:, 1::2]) / math.sqrt(2.0)
    even = (approximation + detail) / math.sqrt(2.0)
    odd = (approximation - detail) / math.sqrt(2.0)
    restored = torch.stack((even, odd), dim=2).flatten(1, 2)[:, : original.shape[1]]
    error = (restored - original).abs()
    return {"full_haar_max_abs": float(error.max()), "full_haar_mean_abs": float(error.mean()), "passed": bool(float(error.max()) <= 1e-5)}
