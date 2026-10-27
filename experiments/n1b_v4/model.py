"""Local temporal residuals after native density development."""
import torch
from Methods.FasterGS4D.Model import Gaussians
from Methods.FasterGS4D.Renderer import FasterGS4DRenderer, extract_settings
from Methods.FasterGS4D.FasterGS4DCudaBackend import diff_rasterize, FusedAdam
from Methods.FasterGS4D.utils import build_4d_rotation
from Optim.adam_utils import prune_param_groups, sort_param_groups


class ResidualGaussians(Gaussians):
    def __init__(self, *args):
        super().__init__(*args)
        self.residuals = torch.nn.ParameterDict()
        self.extra_optimizer = None
        self.point_groups = []
        self.register_buffer('reference_sigma_t', None)
        self.register_buffer('reference_sigma_xyz', None)

    def prune(self, mask):
        super().prune(mask)
        if self.extra_optimizer is not None and self.point_groups:
            updated = prune_param_groups(self.extra_optimizer, ~mask, self.point_groups)
            self.residuals.update(updated)
        for name in ('reference_sigma_t', 'reference_sigma_xyz'):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name)[~mask].contiguous())

    def sort(self, order):
        super().sort(order)
        if self.extra_optimizer is not None and self.point_groups:
            updated = sort_param_groups(self.extra_optimizer, order, self.point_groups)
            self.residuals.update(updated)
        for name in ('reference_sigma_t', 'reference_sigma_xyz'):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name)[order].contiguous())

    @torch.no_grad()
    def activate(self, kind, accel_lr=.01, color_lr=.001):
        assert len(self.residuals) == 0
        specs = []
        n = len(self.spatial_means)
        if kind in ('accel', 'accel_color'):
            specs.append(('accel', (n, 3), accel_lr))
        if kind in ('color', 'accel_color'):
            specs.append(('color', (n, 1, 3), color_lr))
        if kind == 'affine':
            specs.append(('affine', (8, 3, 4), 3e-5))
        assert kind in ('native', 'accel', 'color', 'accel_color', 'affine')
        groups = []
        for name, shape, lr in specs:
            self.residuals[name] = torch.nn.Parameter(torch.zeros(shape, device='cuda'))
            groups.append({'params': [self.residuals[name]], 'lr': lr, 'name': name})
            if name != 'affine':
                self.point_groups.append(name)
        if groups:
            self.extra_optimizer = FusedAdam(groups, lr=0., eps=1e-15)
        if self.point_groups:
            temporal, spatial = [], []
            for i in range(0, n, 65536):
                sl = slice(i, i+65536)
                rotation = build_4d_rotation(self.raw_left_isoclinic_rotations[sl],
                                             self.raw_right_isoclinic_rotations[sl])
                variance = (2 * torch.cat((self.raw_spatial_scales[sl], self.raw_temporal_scales[sl]), -1)).exp()
                tt = (rotation[:, 3].square() * variance).sum(-1, keepdim=True).clamp_min(1e-12)
                xt = (rotation[:, :3] * rotation[:, 3:4] * variance[:, None]).sum(-1)
                xx = (rotation[:, :3].square() * variance[:, None]).sum(-1)
                temporal.append(tt.sqrt().clamp_min(1e-4))
                spatial.append((xx - xt.square()/tt).clamp_min(1e-12).sqrt())
            self.reference_sigma_t = torch.cat(temporal)
            self.reference_sigma_xyz = torch.cat(spatial)
            assert bool(self.reference_sigma_xyz.isfinite().all())


class ResidualRenderer(FasterGS4DRenderer):
    def render_image_training(self, view, update_densification_info, bg_color):
        g = self.model.gaussians
        if not len(g.residuals):
            return super().render_image_training(view, update_densification_info, bg_color)
        means, sh0 = g.spatial_means, g.sh_coefficients_0
        if 'affine' in g.residuals:
            u = max(0., min(1., view.timestamp)) * 7
            left = min(int(u), 6)
            fraction = u - left
            affine = g.residuals['affine'][left]*(1-fraction) + g.residuals['affine'][left+1]*fraction
            means = means + means @ affine[:, :3].T + affine[:, 3]
        if g.point_groups:
            q = ((view.timestamp - g.temporal_means) / g.reference_sigma_t).clamp(-3., 3.)
            if 'accel' in g.residuals:
                means = means + g.reference_sigma_xyz * g.residuals['accel'] * q.square()
            if 'color' in g.residuals:
                sh0 = sh0 + g.residuals['color'] * q[:, :, None]
        info = torch.zeros_like(g.densification_info) if update_densification_info else torch.empty(0)
        image = diff_rasterize(spatial_means=means, temporal_means=g.temporal_means,
            spatial_scales=g.raw_spatial_scales, temporal_scales=g.raw_temporal_scales,
            left_isoclinic_rotations=g.raw_left_isoclinic_rotations,
            right_isoclinic_rotations=g.raw_right_isoclinic_rotations,
            opacities=g.raw_opacities, sh_coefficients_0=sh0,
            sh_coefficients_rest=g.sh_coefficients_rest, densification_info=info,
            rasterizer_settings=extract_settings(view, g.active_sh_bases, bg_color))
        return image, info

    @torch.no_grad()
    def render_image_inference(self, view, to_chw=False):
        image, _ = self.render_image_training(view, False, view.camera.background_color)
        image = image.clamp(0, 1)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}
