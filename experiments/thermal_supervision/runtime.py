"""Fixed-topology post-density model using the real FasterGS4D CUDA backend.

No camera/time gradients, density-control writes, or modifications to the native
submodule. RGB optimizer state and accepted thermal correction are independent.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math

import torch
from torch import Tensor, nn

from .core import Frame, exposure_average, thermal_sh0

RAW_SHAPES = {
    'spatial_means': (3,), 'temporal_means': (1,),
    'spatial_scales': (3,), 'temporal_scales': (1,),
    'left_isoclinic_rotations': (4,), 'right_isoclinic_rotations': (4,),
    'opacities': (1,), 'sh_coefficients_0': (1, 3),
}


@dataclass(frozen=True)
class Camera:
    w2c: Tensor
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float = .1
    far: float = 100.

    def validate(self):
        if self.w2c.shape != (4, 4) or not bool(torch.isfinite(self.w2c).all()):
            raise ValueError('camera requires finite 4x4 w2c')
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy, self.near, self.far)):
            raise ValueError('nonfinite intrinsics')
        if self.width < 1 or self.height < 1 or min(self.fx, self.fy, self.near) <= 0 or self.far <= self.near:
            raise ValueError('invalid camera extent')
        mat = self.w2c.detach().double()
        torch.testing.assert_close(mat[3], mat.new_tensor([0., 0., 0., 1.]), atol=1e-5, rtol=0)
        torch.testing.assert_close(mat[:3, :3] @ mat[:3, :3].T, torch.eye(3, device=mat.device, dtype=mat.dtype), atol=1e-4, rtol=0)
        if not bool(torch.det(mat[:3, :3]) > 0):
            raise ValueError('w2c rotation must preserve handedness')


@dataclass(frozen=True)
class ExposureSample:
    camera: Camera
    time: float  # normalized model time, not an optimizable tensor
    weight: float


@dataclass(frozen=True)
class Observation:
    frame: Frame  # physical timestamps in seconds for synchronization checks
    sensor: str
    samples: tuple[ExposureSample, ...]
    target: Tensor
    weight: Tensor
    noise: Tensor
    gain: float = 1.
    bias: float = 0.

    def validate(self):
        if self.sensor not in ('rgb', 'thermal') or not self.samples:
            raise ValueError('sensor and nonempty exposure samples required')
        channels = 3 if self.sensor == 'rgb' else 1
        h, w = self.samples[0].camera.height, self.samples[0].camera.width
        if self.target.shape != (channels, h, w) or self.weight.shape != (1, h, w):
            raise ValueError('observation dimensions must match native sensor grid')
        for sample in self.samples:
            sample.camera.validate()
            if (sample.camera.height, sample.camera.width) != (h, w):
                raise ValueError('exposure sensor size changed')
            if not math.isfinite(sample.time) or not math.isfinite(sample.weight) or sample.weight < 0:
                raise ValueError('invalid exposure sample')
        if sum(s.weight for s in self.samples) <= 0:
            raise ValueError('empty exposure mass')
        if not math.isfinite(self.gain) or self.gain <= 0 or not math.isfinite(self.bias):
            raise ValueError('invalid fixed sensor calibration')
        for tensor in (self.target, self.weight, self.noise):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('nonfinite observation')
        if bool((self.weight < 0).any()) or not bool(self.weight.max() > 0) or bool((self.noise <= 0).any()):
            raise ValueError('invalid mask/noise')
        torch.broadcast_to(self.noise, self.target.shape)


class ThermalModel(nn.Module):
    def __init__(self, raw: Mapping[str, Tensor], *, active_sh_bases=1, thermal_degree=0,
                 thermal_initial=.5, reference_scale: Tensor | None = None):
        super().__init__()
        n = len(raw['spatial_means'])
        if not n or active_sh_bases not in (1, 4, 9, 16) or thermal_degree not in (0, 1):
            raise ValueError('invalid model dimensions or thermal degree')
        if set(raw) != set(RAW_SHAPES) | {'sh_coefficients_rest'}:
            raise ValueError('native raw parameter keys do not match')
        for name, shape in RAW_SHAPES.items():
            if raw[name].shape != (n, *shape):
                raise ValueError(f'invalid {name} shape')
        rest = raw['sh_coefficients_rest']
        if rest.ndim != 3 or rest.shape[0] != n or rest.shape[2] != 3 or rest.shape[1] < active_sh_bases - 1:
            raise ValueError('invalid higher-order SH shape')
        device = raw['spatial_means'].device
        if any(v.device != device or not bool(torch.isfinite(v).all()) for v in raw.values()):
            raise ValueError('all native tensors must be finite and on one device')
        converted = {k: v.detach().float().clone() for k, v in raw.items()}
        if any(not bool(torch.isfinite(v).all()) for v in converted.values()):
            raise ValueError('native tensor conversion overflow')
        self.raw = nn.ParameterDict({k: nn.Parameter(v) for k, v in converted.items()})
        self.active_sh_bases = active_sh_bases
        self.thermal_degree = thermal_degree
        if not math.isfinite(thermal_initial) or thermal_initial <= 0:
            raise ValueError('thermal initialization must be positive')
        h = torch.zeros((n, thermal_degree + 1), device=device)
        h[:, 0] = math.log(math.expm1(thermal_initial))
        self.thermal = nn.Parameter(h)
        self.register_buffer('thermal_delta', torch.zeros((n, 3), device=device))
        scale = raw['spatial_scales'].detach().exp() if reference_scale is None else reference_scale.detach()
        if scale.shape != (n, 3) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError('invalid reference scales')
        scale = scale.float().clone()
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError('reference scale conversion overflow or underflow')
        self.register_buffer('reference_scale', scale)
        self.register_buffer('point_ids', torch.arange(n, device=device))
        self.version = 0
        self.render_calls = 0

    def render(self, observation: Observation, *, delta: Tensor | None = None,
               thermal: Tensor | None = None, train_raw=False, support: Tensor | None = None) -> Tensor:
        """Render an entire fixed exposure with independent sensor geometry.

        train_raw=True is only for ordinary RGB steps. Thermal graphs always
        detach every original tensor. Temporary leaves supply G/A gradients.
        """
        if self.thermal_delta.device.type != 'cuda':
            raise RuntimeError('real FasterGS4D rendering requires CUDA')
        if train_raw and observation.sensor != 'rgb':
            raise ValueError('thermal cannot directly update native parameters')
        from FasterGS4DCudaBackend.torch_bindings.rasterization import diff_rasterize, RasterizerSettings
        raw = {k: v if train_raw else v.detach() for k, v in self.raw.items()}
        offset = self.thermal_delta.detach() if delta is None else delta
        head = self.thermal.detach() if thermal is None else thermal
        if offset.shape != self.thermal_delta.shape or head.shape != self.thermal.shape:
            raise ValueError('candidate tensors do not match topology')
        raw['spatial_means'] = raw['spatial_means'] + offset
        values = []
        for sample in observation.samples:
            camera = sample.camera
            w2c = camera.w2c.detach().to(device=offset.device, dtype=torch.float32).contiguous()
            position = torch.linalg.inv(w2c)[:3, 3].contiguous()
            args = dict(raw)
            bases = self.active_sh_bases
            if support is not None:
                if support.shape != (len(head),):
                    raise ValueError('one support indicator per primitive required')
                args['sh_coefficients_0'] = ((support[:, None, None].float().expand(-1, 1, 3)-.5)/.28209479177387814).contiguous()
                args['sh_coefficients_rest'] = torch.zeros_like(raw['sh_coefficients_rest'])
                bases = 1
            elif observation.sensor == 'thermal':
                time = float(sample.time)
                log_h = head[:, :1]
                if self.thermal_degree:
                    log_h = log_h + head[:, 1:2] * time
                args['sh_coefficients_0'] = thermal_sh0(log_h)
                args['sh_coefficients_rest'] = torch.zeros_like(raw['sh_coefficients_rest'])
                bases = 1
            settings = RasterizerSettings(w2c, position, torch.zeros(3, device=offset.device), bases,
                camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy,
                camera.near, camera.far, float(sample.time))
            image = diff_rasterize(**{k: v.contiguous() for k, v in args.items()},
                densification_info=torch.empty(0, device=offset.device), rasterizer_settings=settings)
            self.render_calls += 1
            values.append(image if observation.sensor == 'rgb' else image.mean(0, keepdim=True))
        prediction = exposure_average(torch.stack(values), offset.new_tensor([s.weight for s in observation.samples]))
        return prediction if support is not None else prediction * observation.gain + observation.bias

    @torch.no_grad()
    def reindex(self, indices: Tensor, optimizers=()):
        """Explicit prune/permutation after transactions, preserving Adam state.

        Growth/duplicate indices are deliberately rejected in this post-density
        implementation. An optimizer cannot be hidden from this call.
        """
        indices = indices.to(self.point_ids.device)
        n = len(self.point_ids)
        if indices.ndim != 1 or indices.dtype != torch.long or not len(indices):
            raise ValueError('nonempty long indices required')
        if len(indices.unique()) != len(indices) or int(indices.min()) < 0 or int(indices.max()) >= n:
            raise ValueError('reindex is prune/permutation, not densification')
        replacements = {}
        owned = set(self.parameters())
        if any(p not in owned for opt in optimizers for group in opt.param_groups for p in group['params']):
            raise ValueError('optimizer contains a foreign parameter')
        for name, old in [*self.raw.items(), ('thermal', self.thermal)]:
            new = nn.Parameter(old.detach()[indices].contiguous())
            replacements[old] = new
            if name == 'thermal':
                self.thermal = new
            else:
                self.raw[name] = new
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for i, old in enumerate(group['params']):
                    if old not in replacements:
                        raise ValueError('optimizer contains a foreign parameter')
                    new = replacements[old]
                    group['params'][i] = new
                    state = optimizer.state.pop(old, {})
                    optimizer.state[new] = {key: val[indices].clone() if isinstance(val, Tensor) and val.shape == old.shape else val
                                            for key, val in state.items()}
        for name in ('thermal_delta', 'reference_scale', 'point_ids'):
            setattr(self, name, getattr(self, name)[indices].contiguous())
        self.version += 1

    def save(self, path: Path, *, optimizers: Mapping[str, torch.optim.Optimizer] | None = None, extra=None):
        payload = {'schema': 1, 'state': self.state_dict(), 'active_sh_bases': self.active_sh_bases,
                   'thermal_degree': self.thermal_degree, 'version': self.version,
                   'optimizers': {k: v.state_dict() for k, v in (optimizers or {}).items()},
                   'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   'extra': extra or {}}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device='cuda'):
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved.get('schema') == 1 and 'state' in saved:
            state = saved['state']
            raw = {k[4:]: v for k, v in state.items() if k.startswith('raw.')}
            model = cls(raw, active_sh_bases=saved['active_sh_bases'], thermal_degree=saved['thermal_degree'])
            model.load_state_dict(state)
            if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
                raise ValueError('nonfinite checkpoint state')
            if bool((model.reference_scale <= 0).any()) or len(model.point_ids.unique()) != len(model.point_ids):
                raise ValueError('invalid checkpoint scales or primitive identities')
            if not isinstance(saved['version'], int) or saved['version'] < 0:
                raise ValueError('invalid checkpoint version')
            model.version = saved['version']
            return model, saved
        if 'raw' not in saved:
            raise ValueError('requires native raw prefix or CTG checkpoint')
        if saved.get('residuals'):
            raise ValueError('N1b dynamic residual checkpoint needs an explicit adapter; cannot silently discard residuals')
        model = cls(saved['raw'], active_sh_bases=(int(saved.get('active_sh_degree', 0)) + 1)**2)
        return model, {'native_prefix': True, 'steps': saved.get('steps'), 'extra': {}}
