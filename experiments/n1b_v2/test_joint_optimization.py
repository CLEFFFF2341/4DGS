"""Regression checks for coordinate units, trainability, and pixel selection."""
import math
import unittest
import torch

from experiments.n1b.scenes import scene
from experiments.n1b.core import render_cuda
from experiments.n1b.narrow_benchmark import fixture, render
from .joint_optimization import JointParameters, spatial_transform
from .train_probe import render_pixels, prediction


class JointOptimizationTests(unittest.TestCase):
    def test_coordinate_units_do_not_change_adam_step(self):
        initial, _, _, _ = scene('static', device='cpu')
        scaled = {k: v.clone() for k, v in initial.items()}
        scaled['spatial_means'] *= 10
        scaled['spatial_scales'] += math.log(10)
        a, b = JointParameters(initial), JointParameters(scaled)
        for state in (a, b):
            optimizer = state.optimizer()
            displacement = state.raw()['spatial_means'] - state.fixed['spatial_means']
            normalized = torch.linalg.solve_triangular(state.transform, displacement[..., None], upper=False)
            loss = (normalized - .2).square().mean() + (state.temporal - .1).square().mean()
            loss.backward(); optimizer.step()
        torch.testing.assert_close(a.position, b.position)
        torch.testing.assert_close(a.temporal, b.temporal)
        torch.testing.assert_close(b.raw()['spatial_means'], 10 * a.raw()['spatial_means'])

    def test_rotated_transform_is_conditional_not_marginal(self):
        raw, _, _, _ = scene('single', device='cpu')
        # x/t rotation by .4: Schur-complement x variance is det(x,t)/var(t).
        sx2 = raw['spatial_scales'][0, 0].exp().square()
        st2 = raw['temporal_scales'][0, 0].exp().square()
        temporal_variance = math.sin(.4)**2 * sx2 + math.cos(.4)**2 * st2
        expected = sx2 * st2 / temporal_variance
        factor = spatial_transform(raw)
        covariance = factor @ factor.transpose(-1, -2)
        torch.testing.assert_close(covariance[0, 0, 0], expected)

    def test_all_geometry_coordinates_train_and_terminal_rollback_restores(self):
        raw, _, _, _ = scene('single', device='cpu')
        params = JointParameters(raw)
        optimizer = params.optimizer()
        before = params.snapshot()
        loss = params.raw()['spatial_means'].sum() + params.temporal.sum()
        loss.backward()
        self.assertTrue(bool((params.position.grad != 0).all()))
        optimizer.step()
        self.assertFalse(torch.equal(params.position, before[0]))
        params.restore(before)
        for p, old in zip(params.parameters(), before):
            self.assertTrue(torch.equal(p, old))
        torch.testing.assert_close(params.raw()['spatial_means'], raw['spatial_means'])

    def test_temporal_only_keeps_geometry_fixed(self):
        raw, _, _, _ = scene('single', device='cpu')
        params = JointParameters(raw, temporal_only=True)
        self.assertEqual(len(params.parameters()), 1)
        self.assertFalse(params.raw()['spatial_means'].requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'native CUDA renderer required')
    def test_roi_matches_archived_render_and_full_frame_selects_edges(self):
        raw, cam, pix, bg = fixture(.002, 0, False)
        selected = render_pixels(raw, cam, pix, bg, .5)
        torch.testing.assert_close(selected, render(raw, cam, bg, .5), rtol=0, atol=0)
        edge_pixels = pix.new_tensor([[.5, .5], [15.5, 8.5], [31.5, 31.5]])
        full = render_cuda(raw, cam, .5, bg)
        torch.testing.assert_close(render_pixels(raw, cam, edge_pixels, bg, .5),
                                   full[:, [0, 8, 31], [0, 15, 31]].T.double(), rtol=0, atol=0)
        raw['temporal_means'].fill_(10)
        result = prediction('partition', raw, cam, edge_pixels, bg)
        torch.testing.assert_close(result, bg.double().expand(3, 3))


if __name__ == '__main__':
    unittest.main()
