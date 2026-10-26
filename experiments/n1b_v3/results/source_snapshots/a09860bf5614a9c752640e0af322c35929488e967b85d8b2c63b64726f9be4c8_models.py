"""Explicit experimental temporal structures on the unchanged native renderer.

These are successors, not new quadrature rules. Extra parameter counts and
coordinates are reported by the experiment harness.
"""
import torch
from experiments.n1b.core import rotation, render_cuda
from experiments.n1b_v2.joint_optimization import JointParameters
from experiments.n1b_v2.exposure import gauss_rule


def velocity(raw):
    r = rotation(raw['left_isoclinic_rotations'], raw['right_isoclinic_rotations'])
    v = (2 * torch.cat((raw['spatial_scales'], raw['temporal_scales']), -1)).exp()
    cov = (r * v[:, None, :]) @ r.transpose(-1, -2)
    return cov[:, :3, 3] / cov[:, 3:4, 3]


class TemporalModel:
    def __init__(self, initial, kind='native', rate=0.001):
        if kind not in ('native', 'anchor', 'warp', 'curve', 'affine_curve', 'affine_spline', 'weighted_edges'):
            raise ValueError(kind)
        self.kind = kind
        self.joint = JointParameters(initial, mode='scale-aware')
        self.opt = self.joint.optimizer(lr=.05, spatial_lr=.01)
        nuisance = [('sh_coefficients_0', .01), ('sh_coefficients_rest', .0005),
                    ('opacities', .01), ('spatial_scales', .003),
                    ('temporal_means', .0005), ('left_isoclinic_rotations', .0002),
                    ('right_isoclinic_rotations', .0002)]
        for key, lr in nuisance:
            self.joint.fixed[key].requires_grad_()
            self.opt.add_param_group({'params': [self.joint.fixed[key]], 'lr': lr})
        self.anchor = initial['temporal_means'].detach().clone()
        # Local fixed pivot preserves initialization exactly and decouples a
        # temporal-center update from a translation of its conditional path.
        shape = {'warp': (2,), 'curve': (3,), 'affine_curve': (3, 4), 'affine_spline': (8, 3, 4)}.get(kind)
        self.extra = None if shape is None else torch.zeros(shape, device='cuda', requires_grad=True)
        if self.extra is not None:
            self.opt.add_param_group({'params': [self.extra], 'lr': rate})

    def base_raw(self):
        raw = self.joint.raw()
        if self.kind == 'anchor':
            raw['spatial_means'] = raw['spatial_means'] + velocity(raw) * (raw['temporal_means'] - self.anchor)
        return raw

    def at(self, raw, t):
        if self.kind in ('native', 'anchor', 'weighted_edges'):
            return raw
        out = dict(raw)
        if self.kind == 'warp':
            # Endpoint-preserving monotone warp on [0,1]: each polynomial's
            # derivative magnitude <= 1 and each coefficient magnitude < .45.
            a, b = .45 * self.extra.tanh()
            delta = a * t * (1-t) + b * t * (1-t) * (2*t-1)
            out['temporal_means'] = raw['temporal_means'] - delta
        else:
            # Global quadratic component. Subtract its uniform-time mean to
            # separate it from an ordinary static translation.
            q = (t-.5)**2 - 1/12
            if self.kind == 'curve':
                displacement = self.extra
            elif self.kind == 'affine_spline':
                # Eight uniformly spaced temporal anchors, linear interpolation.
                # Their shared affine center field has 96 parameters total.
                u = max(0., min(1., t))*7
                left = min(int(u), 6); fraction = u-left
                affine = self.extra[left]*(1-fraction)+self.extra[left+1]*fraction
                displacement = raw['spatial_means'] @ affine[:, :3].T + affine[:, 3]
                q = 1.
            else:
                displacement = raw['spatial_means'] @ self.extra[:, :3].T + self.extra[:, 3]
            out['spatial_means'] = raw['spatial_means'] + q * displacement
        return out

    def predict(self, data, side, index, calls):
        raw = self.base_raw()
        lo, hi = data.bounds[side, index]
        x, w = gauss_rule(calls)
        nodes = zip((lo+hi)/2+x*(hi-lo)/2, w/2)
        if self.kind == 'weighted_edges':
            from .support_schedule import schedule
            nodes, _ = schedule(raw, data.cameras[side], float(lo), float(hi), calls)
        image = torch.zeros((3, data.h, data.w), device='cuda')
        for t, weight in nodes:
            image = image + float(weight) * render_cuda(self.at(raw, float(t)), data.cameras[side], float(t), data.bg)
        return image

    def state(self):
        return {'kind': self.kind,
                'raw': {k: v.detach().clone() for k, v in self.base_raw().items()},
                'extra': None if self.extra is None else self.extra.detach().clone()}

    def parameter_count(self):
        return sum(p.numel() for g in self.opt.param_groups for p in g['params'])


def predict_state(state, data, side, index, calls=32):
    # Reuse the exact rendering transform without constructing an optimizer.
    model = object.__new__(TemporalModel)
    # All native-geometry sampling arms share ordinary Gauss evaluation.
    model.kind = 'native' if state['kind'] == 'weighted_edges' else state['kind']
    model.extra = state['extra']
    model.base_raw = lambda: state['raw']
    return model.predict(data, side, index, calls)
