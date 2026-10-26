"""Native training callbacks remain inherited and unmodified in both arms."""
import hashlib
import json
import time
import torch
from Methods.Base.utils import pre_training_callback, training_callback
from Methods.FasterGS4D.Trainer import FasterGS4DTrainer
from Methods.FasterGS4D.Renderer import FasterGS4DRenderer, extract_settings
from Methods.FasterGS4D.FasterGS4DCudaBackend import diff_rasterize, FusedAdam

KEYS = ('spatial_means', 'temporal_means', 'spatial_scales', 'temporal_scales',
        'left_isoclinic_rotations', 'right_isoclinic_rotations', 'opacities',
        'sh_coefficients_0', 'sh_coefficients_rest')


def raw_state(model):
    return {key: getattr(model.gaussians, '_' + key).detach().cpu() for key in KEYS}


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FieldRenderer(FasterGS4DRenderer):
    def transformed_means(self, timestamp):
        # Exactly the frozen v3 affine_spline transform: eight knots, 96 values.
        u = max(0., min(1., timestamp)) * 7
        left = min(int(u), 6)
        fraction = u - left
        field = self.model.temporal_field
        affine = field[left] * (1 - fraction) + field[left + 1] * fraction
        means = self.model.gaussians.spatial_means
        return means + means @ affine[:, :3].T + affine[:, 3]

    def render_image_training(self, view, update_densification_info, bg_color):
        g = self.model.gaussians
        info = torch.zeros_like(g.densification_info) if update_densification_info else torch.empty(0)
        image = diff_rasterize(
            spatial_means=self.transformed_means(view.timestamp),
            temporal_means=g.temporal_means,
            spatial_scales=g.raw_spatial_scales, temporal_scales=g.raw_temporal_scales,
            left_isoclinic_rotations=g.raw_left_isoclinic_rotations,
            right_isoclinic_rotations=g.raw_right_isoclinic_rotations,
            opacities=g.raw_opacities, sh_coefficients_0=g.sh_coefficients_0,
            sh_coefficients_rest=g.sh_coefficients_rest, densification_info=info,
            rasterizer_settings=extract_settings(view, g.active_sh_bases, bg_color))
        return image, info

    @torch.no_grad()
    def render_image_inference(self, view, to_chw=False):
        assert self.SCALE_MODIFIER == 1 and self.RENDER_DIFFUSE and self.RENDER_SPECULAR
        image, _ = self.render_image_training(view, False, view.camera.background_color)
        image = image.clamp(0, 1)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}


class AuditedNativeTrainer(FasterGS4DTrainer):
    @pre_training_callback(priority=39)
    def audit_initialization(self, _, dataset):
        state = raw_state(self.model)
        self.initial_hash = tensor_hash(state)
        self.initial_count = len(state['spatial_means'])
        self.zero_field_check = None
        if hasattr(self.model, 'temporal_field'):
            view = dataset.train()[0]
            original = FasterGS4DRenderer(self.model)
            means = self.renderer.transformed_means(view.timestamp)
            assert torch.equal(means, self.model.gaussians.spatial_means)
            a, _ = original.render_image_training(view, False, view.camera.background_color)
            b, _ = self.renderer.render_image_training(view, False, view.camera.background_color)
            c, _ = original.render_image_training(view, False, view.camera.background_color)
            ga = torch.autograd.grad(a.mean(), self.model.gaussians.spatial_means)[0]
            gb, gf = torch.autograd.grad(b.mean(), [self.model.gaussians.spatial_means,
                                                  self.model.temporal_field])
            gc = torch.autograd.grad(c.mean(), self.model.gaussians.spatial_means)[0]
            self.zero_field_check = {'image_max_abs': float((a-b).abs().max()),
                'native_repeat_max_abs': float((a-c).abs().max()),
                'image_rmse': float((a-b).square().mean().sqrt()),
                'native_repeat_rmse': float((a-c).square().mean().sqrt()),
                'means_gradient_max_abs': float((ga-gb).abs().max()),
                'means_gradient_relative_l2': float((ga-gb).norm()/ga.norm().clamp_min(1e-12)),
                'native_gradient_repeat_relative_l2': float((ga-gc).norm()/ga.norm().clamp_min(1e-12)),
                'transformed_means_bitwise_equal': True,
                'field_gradient_norm': float(gf.norm()), 'field_gradient_finite': bool(gf.isfinite().all())}
            print(json.dumps({'zero_field_check': self.zero_field_check}), flush=True)
            # Identical native renders themselves differ on overlapping DSEC
            # initialization points. Bound the discrepancy against that control.
            check = self.zero_field_check
            assert check['image_rmse'] <= 3 * check['native_repeat_rmse'] + 1e-7
            assert check['means_gradient_relative_l2'] <= 3 * check['native_gradient_repeat_relative_l2'] + 2e-5
            assert bool(gf.isfinite().all()) and float(gf.norm()) > 0
        self.audit_rows = []
        torch.cuda.synchronize()
        self.training_started = time.perf_counter()
        (self.output_directory / 'initial_audit.json').write_text(json.dumps({
            'sha256': self.initial_hash, 'count': self.initial_count,
            'zero_field_check': self.zero_field_check}, indent=2))

    @training_callback(priority=2, iteration_stride=1000)
    def audit_progress(self, iteration, dataset):
        torch.cuda.synchronize()
        row = {'iteration': iteration + 1,
               'count': len(self.model.gaussians.spatial_means),
               'active_sh_degree': self.model.gaussians.active_sh_degree,
               'cuda_allocated': torch.cuda.memory_allocated(),
               'cuda_reserved': torch.cuda.memory_reserved(),
               'requires_empty_cache': self.requires_empty_cache,
               'seconds': time.perf_counter() - self.training_started}
        self.audit_rows.append(row)
        with (self.output_directory / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


class FieldTrainer(AuditedNativeTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Match official construction order: the allocator capability probe in
        # the parent constructor must run before any extra CUDA allocations.
        self.model.temporal_field = torch.nn.Parameter(torch.zeros((8, 3, 4), device='cuda'))
        # Density-control utilities assume all native optimizer groups have N
        # rows. Keep the global 96 parameters in a separate identical optimizer.
        self.field_optimizer = FusedAdam([self.model.temporal_field], lr=3e-5, eps=1e-15)

    @training_callback(priority=79)
    def step_temporal_field(self, *_):
        self.field_optimizer.step()
        self.field_optimizer.zero_grad(set_to_none=True)


assert AuditedNativeTrainer.training_iteration is FasterGS4DTrainer.training_iteration
assert FieldTrainer.training_iteration is FasterGS4DTrainer.training_iteration
