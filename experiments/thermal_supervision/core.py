"""CPU-testable primitives for a proposed thermal geometry transfer gate.

This is not a FasterGS4D trainer or a reconstruction benchmark. The caller must
provide calibrated, exposure-integrated renders and fixed observation masks.
All proposals are functional: neither model tensors nor optimizer state mutate.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import torch
from torch import Tensor


def _finite(value: Tensor, name: str) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'nonfinite {name}')
    return value


def _normalized_weights(weight: Tensor) -> Tensor:
    # Dividing by the maximum first avoids overflow in the normalization sum.
    if weight.dtype in (torch.float16, torch.bfloat16):
        weight = weight.float()
    scaled = weight / weight.max()
    return _finite(scaled / scaled.sum(), 'normalized weights')


def exposure_average(linear_samples: Tensor, weights: Tensor) -> Tensor:
    """Average K linear sensor renders; apply camera response AFTER this call.

    Sample camera poses and timestamps are externally calibrated constants.
    Integrating gamma-coded RGB or pretending thermal exposure is zero is not
    justified by this utility. Thermal PSF/lag must be handled by the caller.
    """
    if linear_samples.ndim < 2 or weights.ndim != 1:
        raise ValueError('expected [K,...] samples and [K] weights')
    if len(weights) != len(linear_samples) or not len(weights):
        raise ValueError('one weight per exposure sample is required')
    if not bool(torch.isfinite(linear_samples).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError('nonfinite exposure inputs')
    if not linear_samples.is_floating_point():
        raise ValueError('exposure samples must be floating point')
    cast_weights = _finite(weights.detach().to(linear_samples), 'cast exposure weights')
    if bool((cast_weights < 0).any()) or not bool(cast_weights.max() > 0):
        raise ValueError('exposure weights must be nonnegative with positive mass')
    normalized = _normalized_weights(cast_weights)
    return _finite((linear_samples * normalized.reshape((-1,) + (1,) * (linear_samples.ndim - 1))).sum(0),
                   'exposure result')


def _inputs(pred: Tensor, target: Tensor, weight: Tensor, noise: Tensor):
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError('pred and target must have matching [C,H,W] shape')
    if weight.shape != (1, *pred.shape[-2:]):
        raise ValueError('fixed pixel weights must have shape [1,H,W]')
    if not pred.is_floating_point():
        raise ValueError('sensor predictions must be floating point')
    # A 256x256 all-ones mask already overflows an FP16 reduction. Preserve
    # gradients through promotion, but perform sensor arithmetic at least FP32.
    if pred.dtype in (torch.float16, torch.bfloat16):
        pred = pred.float()
    target = target.detach().to(pred)
    weight = torch.broadcast_to(weight.detach().to(pred), pred.shape)
    noise = torch.broadcast_to(noise.detach().to(pred), pred.shape)
    if not all(bool(torch.isfinite(x).all()) for x in (pred, target, weight, noise)):
        raise ValueError('nonfinite evidence')
    if bool((weight < 0).any()) or not bool(weight.max() > 0) or bool((noise <= 0).any()):
        raise ValueError('positive support and positive sensor noise are required')
    return pred, target, weight, noise


def sensor_loss(pred: Tensor, target: Tensor, weight: Tensor, noise: Tensor) -> Tensor:
    """Noise-standardized smooth L1; observations/weights/noise cannot learn."""
    pred, target, weight, noise = _inputs(pred, target, weight, noise)
    residual = _finite((pred - target) / noise, 'standardized residual')
    robust = torch.hypot(residual, torch.ones_like(residual)) - 1
    return _finite((_normalized_weights(weight) * robust).sum(), 'sensor loss')


def observation_change(before: Tensor, after: Tensor, weight: Tensor, noise: Tensor) -> Tensor:
    before, after, weight, noise = _inputs(before, after, weight, noise)
    residual = _finite((before - after) / noise, 'observation difference')
    scale = residual.abs().max().clamp_min(torch.finfo(residual.dtype).tiny)
    result = scale * (_normalized_weights(weight) * (residual / scale).square()).sum().sqrt()
    return _finite(result, 'observation change')


def thermal_sh0(log_radiance: Tensor) -> Tensor:
    """Pack one positive IR radiance into native [N,1,3] SH0 coefficients.

    All channels represent the same scalar; reduce the returned rendered image
    by channel MEAN, not sum. Set active_sh_bases=1 and use separate IR camera.
    Native higher SH buffers may still need allocated shape; never use RGB SH.
    """
    if log_radiance.ndim != 2 or log_radiance.shape[1] != 1:
        raise ValueError('thermal log radiance must have shape [N,1]')
    radiance = torch.nn.functional.softplus(_finite(log_radiance, 'thermal parameter'))
    result = ((radiance - 0.5) / 0.28209479177387814).unsqueeze(1).expand(-1, -1, 3).contiguous()
    return _finite(result, 'thermal SH0')


def route_tensors(tensors: Mapping[str, Tensor], active: Sequence[str]) -> dict[str, Tensor]:
    """Expose gradients only to an explicit proposal subset, without hooks."""
    unknown = set(active) - tensors.keys()
    if unknown:
        raise ValueError(f'unknown active tensors: {sorted(unknown)}')
    return {key: value if key in active else value.detach() for key, value in tensors.items()}


def bounded_proposal(value: Tensor, gradient: Tensor, reference_scale: Tensor,
                     radius: float) -> Tensor:
    """One source-only, scale-normalized SGD proposal with per-row norm bound.

    Returned tensor is detached. It does not change value, gradients, Adam
    moments, densification buffers, RNG, or accepted thermal residual state.
    The caller additionally enforces a CUMULATIVE displacement cap.
    """
    if value.shape != gradient.shape or value.ndim != 2:
        raise ValueError('value and gradient must have the same [N,D] shape')
    if not isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    scale = torch.broadcast_to(reference_scale.detach(), value.shape)
    if not all(bool(torch.isfinite(x).all()) for x in (value, gradient, scale)) or bool((scale <= 0).any()):
        raise ValueError('invalid proposal tensors')
    direction = _finite(-gradient.detach() * scale, 'proposal direction')
    maximum = direction.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    scaled = direction / maximum
    direction = scaled / torch.maximum(scaled.norm(dim=-1, keepdim=True), maximum.reciprocal())
    step = _finite(radius * scale * direction, 'proposal step')
    return _finite(value.detach() + step, 'proposal result')


@dataclass(frozen=True)
class Frame:
    view_id: str
    viewpoint_id: str
    timestamp: float
    split: str = 'train'
    calibrated: bool = False


@dataclass(frozen=True)
class ThermalEvidence:
    frame: Frame
    target: Tensor
    before: Tensor
    geometry: Tensor
    appearance: Tensor
    weight: Tensor  # fixed support common to all three renders
    noise: Tensor


@dataclass(frozen=True)
class RGBEvidence:
    frame: Frame
    target: Tensor
    before: Tensor
    geometry: Tensor  # exposure-integrated geometry trial, NOT a sharp render
    weight: Tensor  # observed reliable RGB, fixed for the whole transaction
    noise: Tensor


@dataclass(frozen=True)
class GateConfig:
    min_pixels: int = 64
    min_source_gain: float = 1e-4
    min_witness_gain: float = 1e-4
    appearance_advantage_margin: float = 1e-4
    source_gain_relative_tolerance: float = 0.2
    max_source_change: float = 2.0
    max_rgb_loss_increase: float = 0.0
    max_rgb_anchor_increase: float = 0.0
    min_witness_views: int = 1
    synchronization_tolerance: float = 1e-4


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    source_geometry_gain: float = 0.0
    source_appearance_gain: float = 0.0
    witness_advantages: tuple[float, ...] = ()


@torch.no_grad()
def decide_geometry_transfer(source: ThermalEvidence, witnesses: Sequence[ThermalEvidence],
                             rgb_guards: Sequence[RGBEvidence], *,
                             config: GateConfig = GateConfig(), static_scene: bool = True,
                             rgb_anchor_losses: Sequence[float] | None = None) -> GateDecision:
    """Accept only matched-fit geometry gains that transfer to OTHER views.

    These are optimization views, not statistically held-out validation/test
    views. This gate is a finite-evidence heuristic, not a geometry proof.
    rgb_anchor_losses are fixed-window pre-transfer reference losses, aligned
    with rgb_guards, to prevent repeated small tolerated losses accumulating.
    Caller is responsible for source-only candidate search and valid overlapping
    observations. This function does not build correspondences or calibrate data.
    """
    def reject(reason):
        return GateDecision(False, reason)

    thresholds = (config.min_source_gain, config.min_witness_gain,
                  config.appearance_advantage_margin, config.max_rgb_loss_increase,
                  config.max_rgb_anchor_increase, config.synchronization_tolerance)
    if (any(not isfinite(x) or x < 0 for x in thresholds)
            or config.min_pixels < 1 or config.min_witness_views < 1
            or not 0 <= config.source_gain_relative_tolerance < 1
            or not isfinite(config.max_source_change) or config.max_source_change <= 0):
        raise ValueError('invalid gate configuration')
    if len(witnesses) < config.min_witness_views or not rgb_guards:
        return reject('insufficient independent thermal or RGB evidence')
    if rgb_anchor_losses is None or len(rgb_anchor_losses) != len(rgb_guards):
        return reject('missing fixed-window RGB anchor')
    thermal_frames = [source.frame] + [w.frame for w in witnesses]
    all_frames = thermal_frames + [g.frame for g in rgb_guards]
    if any(f.split != 'train' for f in all_frames):
        return reject('validation/test view cannot control optimization')
    if any(not f.calibrated or not isfinite(f.timestamp) for f in all_frames):
        return reject('invalid camera/time calibration')
    if len({f.view_id for f in thermal_frames}) != len(thermal_frames):
        return reject('duplicated thermal observation')
    if len({f.viewpoint_id for f in thermal_frames}) != len(thermal_frames):
        return reject('thermal witnesses need distinct viewpoints')
    if not static_scene and any(abs(f.timestamp - source.frame.timestamp) > config.synchronization_tolerance for f in all_frames):
        return reject('dynamic comparison requires synchronized observations')
    evidence = [source, *witnesses, *rgb_guards]
    if any(int((e.weight.detach() > 0).sum()) < config.min_pixels for e in evidence):
        return reject('insufficient fixed observation support')
    try:
        def gains(e):
            before = sensor_loss(e.before, e.target, e.weight, e.noise)
            return (float(before - sensor_loss(e.geometry, e.target, e.weight, e.noise)),
                    float(before - sensor_loss(e.appearance, e.target, e.weight, e.noise)))
        source_g, source_a = gains(source)
        if not all(isfinite(x) for x in (source_g, source_a)):
            return reject('nonfinite source gains')
        if min(source_g, source_a) <= config.min_source_gain:
            return reject('one counterfactual does not explain source residual')
        if abs(source_g - source_a) > config.source_gain_relative_tolerance * max(source_g, source_a):
            return reject('unmatched source fit; geometry attribution is inconclusive')
        for prediction in (source.geometry, source.appearance):
            if float(observation_change(source.before, prediction, source.weight, source.noise)) > config.max_source_change:
                return reject('source observation change exceeds trust budget')
        advantages = []
        for witness in witnesses:
            gain_g, gain_a = gains(witness)
            if not all(isfinite(x) for x in (gain_g, gain_a, gain_g - gain_a)):
                return reject('nonfinite witness gains')
            advantages.append(gain_g - gain_a)
            if gain_g <= config.min_witness_gain or gain_g - gain_a <= config.appearance_advantage_margin:
                return reject('thermal geometry does not beat appearance on every witness')
        for guard, anchor in zip(rgb_guards, rgb_anchor_losses):
            if not isfinite(anchor) or anchor < 0:
                return reject('invalid fixed-window RGB anchor')
            before = float(sensor_loss(guard.before, guard.target, guard.weight, guard.noise))
            after = float(sensor_loss(guard.geometry, guard.target, guard.weight, guard.noise))
            if after > before + config.max_rgb_loss_increase:
                return reject('geometry trial damages reliable RGB observation')
            if after > anchor + config.max_rgb_anchor_increase:
                return reject('geometry trial exceeds fixed-window RGB damage budget')
    except (ValueError, RuntimeError) as exc:
        return reject(f'invalid rendered evidence: {exc}')
    return GateDecision(True, 'cross-view thermal advantage with RGB guard',
                        source_g, source_a, tuple(advantages))
