import unittest
import torch
from experiments.n1b.core import render_cuda
from experiments.n1b.narrow_benchmark import fixture
from experiments.n1b.scenes import scene
from .exposure import Exposure, predict_exposure, stationary_reference
from .train_probe import prediction
from .joint_probe import reference_image


class ExposureTests(unittest.TestCase):
    def test_unknown_or_invalid_duration_is_rejected(self):
        for lo, hi in [(0, 0), (1, 0), (0, float('inf')), (float('nan'), 1)]:
            with self.assertRaises(ValueError):
                Exposure(lo, hi)

    @unittest.skipUnless(torch.cuda.is_available(), 'native renderer required')
    def test_legacy_shutter_and_time_translation_equivariance(self):
        raw, cam, pix, bg = fixture(.002, .003, False)
        shutter = Exposure(.3, .7)
        full = predict_exposure(raw, cam, bg, shutter)
        old = prediction('partition', raw, cam, pix, bg)
        torch.testing.assert_close(full[:, 12:20, 12:20].permute(1, 2, 0).reshape(-1, 3), old,
                                   rtol=1e-10, atol=1e-12)
        shifted = {k: v.clone() for k, v in raw.items()}
        shifted['temporal_means'] += .125
        other = predict_exposure(shifted, cam, bg, Exposure(.425, .825))
        torch.testing.assert_close(other, full, rtol=2e-4, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'native renderer required')
    def test_empty_observation_has_zero_gradient_and_exact_background(self):
        raw, cam, pix, bg = fixture(.002, 0, False)
        raw['spatial_means'].requires_grad_(); raw['temporal_scales'].requires_grad_()
        value = predict_exposure(raw, cam, bg, Exposure(.1, .2))
        value.sum().backward()
        torch.testing.assert_close(value, bg.double()[:, None, None].expand_as(value))
        self.assertEqual(float(raw['spatial_means'].grad.abs().max()), 0.)
        self.assertEqual(float(raw['temporal_scales'].grad.abs().max()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'native renderer required')
    def test_merged_native_timestamps_preserve_legacy_reference(self):
        raw, cam, _, bg = fixture(.002, .003, False)
        yy, xx = torch.meshgrid(torch.arange(32, device=bg.device), torch.arange(32, device=bg.device), indexing='ij')
        pixels = torch.stack((xx.flatten()+.5, yy.flatten()+.5), -1)
        old = reference_image(raw, cam, pixels, bg, 8)
        new = stationary_reference(raw, cam, bg, Exposure(.3, .7), 8)
        torch.testing.assert_close(new.permute(1, 2, 0).reshape(-1, 3), old, rtol=1e-10, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), 'native renderer required')
    def test_arbitrary_duration_preserves_constant_image(self):
        raw, cam, _, bg = scene('static')
        target = render_cuda(raw, cam, .5, bg).double()
        for shutter in [Exposure(.1, .9), Exposure(.5, .502)]:
            for method in ['partition', 'gauss']:
                value = predict_exposure(raw, cam, bg, shutter, 8, method)
                torch.testing.assert_close(value, target, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(stationary_reference(raw, cam, bg, shutter), target,
                                       rtol=1e-10, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
