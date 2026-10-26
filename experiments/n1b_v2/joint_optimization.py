"""Scale-aware coordinates for the small-model joint fitting probe.

The spatial transform is frozen at initialization and uses only model covariance,
never a target position. All three position coordinates remain trainable. This is
an optimizer preconditioner, not a geometry prior or an identifiability guarantee.
"""
import torch
from experiments.n1b.core import rotation


@torch.no_grad()
def spatial_transform(raw):
    R = rotation(raw['left_isoclinic_rotations'], raw['right_isoclinic_rotations'])
    scales = torch.cat((raw['spatial_scales'], raw['temporal_scales']), -1)
    covariance = (R * (2 * scales).exp()[:, None, :]) @ R.transpose(-1, -2)
    cross = covariance[:, :3, 3]
    conditional = covariance[:, :3, :3] - cross[:, :, None] * cross[:, None, :] / covariance[:, 3:4, 3:4]
    return torch.linalg.cholesky(conditional)


class JointParameters:
    def __init__(self, initial, mode='scale-aware', temporal_only=False):
        if mode not in ('legacy', 'scale-aware'):
            raise ValueError('unknown joint coordinate mode')
        self.mode = mode
        self.temporal_only = temporal_only
        self.fixed = {k: v.detach().clone() for k, v in initial.items()}
        self.temporal = self.fixed['temporal_scales'].clone().requires_grad_()
        self.transform = spatial_transform(self.fixed) if mode == 'scale-aware' else None
        self.position = (torch.zeros_like(self.fixed['spatial_means']) if mode == 'scale-aware'
                         else self.fixed['spatial_means'].clone()).requires_grad_(not temporal_only)

    def raw(self):
        raw = dict(self.fixed)
        raw['temporal_scales'] = self.temporal
        if not self.temporal_only:
            raw['spatial_means'] = (self.fixed['spatial_means'] +
                                   (self.transform @ self.position[..., None]).squeeze(-1)
                                   if self.mode == 'scale-aware' else self.position)
        return raw

    def parameters(self):
        return [self.temporal] if self.temporal_only else [self.position, self.temporal]

    def optimizer(self, lr=.01, spatial_lr=None):
        groups = [{'params': [self.temporal], 'lr': lr}]
        if not self.temporal_only:
            groups.append({'params': [self.position], 'lr': lr if spatial_lr is None else spatial_lr})
        return torch.optim.Adam(groups)

    @torch.no_grad()
    def snapshot(self):
        return [p.detach().clone() for p in self.parameters()]

    @torch.no_grad()
    def restore(self, snapshot):
        # Terminal budget rollback only: the optimizer is not resumed afterward.
        for p, old in zip(self.parameters(), snapshot):
            p.copy_(old)
