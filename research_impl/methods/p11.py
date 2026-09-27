from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from diff_gaussian_rasterization_df import GaussianRasterizationSettings, GaussianRasterizer

from .p01 import quotas


@dataclass(frozen=True)
class GateSelection:
    static_indices: np.ndarray
    dynamic_indices: np.ndarray
    probabilities: np.ndarray
    diagnostics: dict[str, Any]


def initial_logits(scores: np.ndarray) -> torch.Tensor:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("P02.E must be non-empty and finite")
    normalized = (values - values.mean()) / (values.std() + 1e-8)
    base = math.log(0.9 / 0.1)
    # Keep the registered 1e-3 score perturbation in FP64 so close P02.E
    # values do not collapse into artificial FP32 ties before stable top-K.
    return torch.as_tensor(base + 1e-3 * normalized, dtype=torch.float64, device="cuda")


def _stable_group_topk(values: torch.Tensor, stable_rows: np.ndarray, count: int) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    stable_order_np = np.argsort(np.asarray(stable_rows), kind="stable")
    stable_order = torch.as_tensor(stable_order_np, dtype=torch.long, device=values.device)
    ordered = values.detach().index_select(0, stable_order)
    ranking = torch.argsort(ordered, descending=True, stable=True)
    return stable_order.index_select(0, ranking[:count])


def hard_group_mask(probabilities: torch.Tensor, adapter: Any, budget_fraction: float) -> torch.Tensor:
    total, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, budget_fraction)
    if probabilities.numel() != adapter.total_count:
        raise ValueError("Gate vector does not match model point count")
    result = torch.zeros_like(probabilities)
    static = _stable_group_topk(probabilities[: adapter.static_count], adapter.static_rows, keep_static)
    dynamic = _stable_group_topk(
        probabilities[adapter.static_count :], adapter.dynamic_rows, keep_dynamic
    )
    result[static] = 1.0
    result[adapter.static_count + dynamic] = 1.0
    if int(result.sum().item()) != total:
        raise AssertionError("Hard gate violates the exact total budget")
    return result


def selection_from_probabilities(probabilities: torch.Tensor, adapter: Any, budget_fraction: float) -> GateSelection:
    hard = hard_group_mask(probabilities, adapter, budget_fraction)
    static = torch.nonzero(hard[: adapter.static_count], as_tuple=False).flatten().cpu().numpy().astype(np.int64)
    dynamic = (
        torch.nonzero(hard[adapter.static_count :], as_tuple=False).flatten().cpu().numpy().astype(np.int64)
    )
    total, keep_static, keep_dynamic = quotas(adapter.static_count, adapter.dynamic_count, budget_fraction)
    return GateSelection(
        static_indices=static,
        dynamic_indices=dynamic,
        probabilities=probabilities.detach().cpu().numpy(),
        diagnostics={
            "requested_total": total,
            "actual_total": int(static.size + dynamic.size),
            "static_quota": keep_static,
            "dynamic_quota": keep_dynamic,
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "probability_mean": float(probabilities.mean()),
        },
    )


def two_point_recovery_toy() -> dict[str, Any]:
    device = "cuda"
    settings = GaussianRasterizationSettings(
        image_height=16,
        image_width=16,
        tanfovx=1.0,
        tanfovy=1.0,
        kernel_size=0.1,
        subpixel_offset=torch.zeros((16, 16, 2), dtype=torch.float32, device=device),
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, dtype=torch.float32, device=device),
        projmatrix=torch.eye(4, dtype=torch.float32, device=device),
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=device),
        prefiltered=False,
        min_depth=0.2,
        max_depth=100.0,
        debug=False,
    )
    rasterizer = GaussianRasterizer(settings)
    means = torch.tensor([[-0.05, 0.0, 1.0], [0.05, 0.0, 1.0]], device=device)
    means2d = torch.zeros_like(means, requires_grad=True)
    flow = torch.zeros_like(means)
    opacity = torch.full((2, 1), 0.8, device=device)
    gate = torch.tensor([0.0, 1.0], device=device, requires_grad=True)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    scales = torch.full((2, 3), 0.1, device=device)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
    image = rasterizer(
        means,
        means2d,
        flow,
        opacity,
        mask_gate=gate,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
    )[0]
    image[0].mean().backward()
    result = {
        "image_sum": float(image.sum()),
        "gate_gradients": gate.grad.detach().cpu().tolist(),
        "masked_point_has_recovery_gradient": bool(abs(float(gate.grad[0])) > 0),
        "unmasked_green_has_zero_red_loss_gradient": bool(float(gate.grad[1]) == 0.0),
    }
    result["passed"] = result["masked_point_has_recovery_gradient"]
    return result
