"""Contract checks only: small CPU tensors, no model fitting or trainer."""
import unittest
from dataclasses import replace

import torch

from .core import (Frame, ThermalEvidence, RGBEvidence, GateConfig, sensor_loss,
                   exposure_average, thermal_sh0, route_tensors, bounded_proposal,
                   decide_geometry_transfer)


def plane(x):
    return torch.full((1, 8, 8), float(x), dtype=torch.float64)


def example():
    # Matched source fit, but geometry transfers better to a different view.
    source = ThermalEvidence(Frame('s', 'pose_s', 0., calibrated=True), plane(0), plane(1),
                             plane(.5), plane(.5), plane(1), plane(1))
    witness = ThermalEvidence(Frame('w', 'pose_w', 0., calibrated=True), plane(0), plane(1),
                              plane(.4), plane(.9), plane(1), plane(1))
    guard = RGBEvidence(Frame('rgb_w', 'pose_w', 0., calibrated=True), plane(0), plane(.1),
                        plane(.1), plane(1), plane(1))
    anchor = float(sensor_loss(guard.before, guard.target, guard.weight, guard.noise))
    return source, witness, guard, anchor


class SensorTests(unittest.TestCase):
    def test_linear_exposure_value_and_gradient(self):
        samples = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([1., 3.], dtype=torch.float64, requires_grad=True)
        result = exposure_average(samples, weights)
        self.assertEqual(float(result.detach()), 1.5)
        result.sum().backward()
        torch.testing.assert_close(samples.grad, torch.tensor([[.25], [.75]], dtype=torch.float64))
        self.assertIsNone(weights.grad)

    def test_exposure_rejects_empty_negative_nonfinite(self):
        for weights in (torch.tensor([0., 0.]), torch.tensor([-1., 2.]), torch.tensor([float('nan'), 1.])):
            with self.assertRaises(ValueError):
                exposure_average(torch.ones((2, 1)), weights)

    def test_extreme_weights_normalize_and_cast_overflow_rejects(self):
        value = exposure_average(torch.ones((2, 1)), torch.full((2,), 3e38))
        torch.testing.assert_close(value, torch.ones(1))
        with self.assertRaises(ValueError):
            exposure_average(torch.ones((2, 1), dtype=torch.float16), torch.full((2,), 3e38))

    def test_noise_cannot_increase_tensor_rank(self):
        with self.assertRaises(RuntimeError):
            sensor_loss(plane(0), plane(0), plane(1), torch.ones((2, 1, 1, 1)))

    def test_half_precision_mask_reduction_preserves_nonzero_loss(self):
        ones = torch.ones((1, 256, 256), dtype=torch.float16, requires_grad=True)
        zeros = torch.zeros_like(ones)
        actual = sensor_loss(ones, zeros, ones.detach(), ones.detach())
        reference = sensor_loss(ones.float(), zeros.float(), ones.float().detach(), ones.float().detach())
        self.assertEqual(actual.dtype, torch.float32)
        self.assertGreater(float(actual.detach()), .4)
        torch.testing.assert_close(actual, reference)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(ones.grad).all()))
        self.assertTrue(bool((ones.grad > 0).all()))

    def test_proposal_rejects_intermediate_overflow(self):
        with self.assertRaises(ValueError):
            bounded_proposal(torch.zeros((1, 3)), torch.full((1, 3), 3e38), torch.ones((1, 3)) * 2, .1)

    def test_sensor_constants_do_not_receive_gradients(self):
        prediction = plane(.3).requires_grad_()
        target, weight, noise = [plane(v).requires_grad_() for v in (.1, 1, .05)]
        sensor_loss(prediction, target, weight, noise).backward()
        self.assertTrue(bool((prediction.grad > 0).all()))
        self.assertTrue(all(x.grad is None for x in (target, weight, noise)))

    def test_thermal_channels_have_unit_not_triple_gradient(self):
        log_h = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
        coefficients = thermal_sh0(log_h)
        self.assertEqual(coefficients.shape, (2, 1, 3))
        scalar = (coefficients * 0.28209479177387814 + .5).mean(-1)
        scalar.sum().backward()
        torch.testing.assert_close(log_h.grad, torch.full_like(log_h, .5))

    def test_thermal_gradient_isolated_from_rgb_and_opacity(self):
        state = {key: torch.tensor(1., requires_grad=True) for key in ('xyz', 'rgb', 'ir', 'opacity')}
        routed = route_tensors(state, ('xyz',))
        (routed['xyz'] * routed['ir'] * routed['opacity'] + routed['rgb']).backward()
        self.assertEqual(float(state['xyz'].grad), 1.)
        self.assertTrue(all(state[key].grad is None for key in ('rgb', 'ir', 'opacity')))

    def test_proposal_is_bounded_and_does_not_mutate_state(self):
        value = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
        gradient = torch.tensor([[100., -20., 7.], [0., 0., 0.]], dtype=torch.float64)
        scale = torch.tensor([[1., 2., .5]], dtype=torch.float64)
        before = value.detach().clone()
        proposal = bounded_proposal(value, gradient, scale, .05)
        self.assertFalse(proposal.requires_grad)
        torch.testing.assert_close(value.detach(), before)
        self.assertTrue(bool((((proposal - before) / scale).norm(dim=-1) <= .05 + 1e-12).all()))
        torch.testing.assert_close(proposal[1], before[1])


class GateTests(unittest.TestCase):
    def run_gate(self, source=None, witness=None, guard=None, anchor=None, **kwargs):
        s, w, g, a = example()
        return decide_geometry_transfer(source or s, [witness or w], [guard or g],
                                        rgb_anchor_losses=[a if anchor is None else anchor], **kwargs)

    def test_accepts_cross_view_gain_without_mutating_evidence(self):
        s, w, g, a = example()
        tensors = [s.before.clone(), s.geometry.clone(), w.before.clone()]
        decision = self.run_gate(source=s, witness=w, guard=g, anchor=a)
        self.assertTrue(decision.accepted, decision.reason)
        for actual, expected in zip((s.before, s.geometry, w.before), tensors):
            torch.testing.assert_close(actual, expected)

    def test_rejects_thermal_appearance_explanation(self):
        _, witness, _, _ = example()
        result = self.run_gate(witness=replace(witness, appearance=plane(.2)))
        self.assertFalse(result.accepted)
        self.assertIn('does not beat appearance', result.reason)

    def test_rejects_matched_fit_failure(self):
        source, _, _, _ = example()
        result = self.run_gate(source=replace(source, appearance=plane(.99)))
        self.assertFalse(result.accepted)
        self.assertIn('unmatched', result.reason)

    def test_rejects_rgb_damage(self):
        _, _, guard, _ = example()
        result = self.run_gate(guard=replace(guard, geometry=plane(.2)))
        self.assertFalse(result.accepted)
        self.assertIn('damages reliable RGB', result.reason)

    def test_cumulative_anchor_blocks_repeated_small_damage(self):
        _, _, guard, _ = example()
        result = self.run_gate(guard=replace(guard, geometry=plane(.2)), anchor=0.,
                               config=GateConfig(max_rgb_loss_increase=.1))
        self.assertFalse(result.accepted)
        self.assertIn('fixed-window', result.reason)

    def test_rejects_test_or_validation_leakage(self):
        source, witness, guard, _ = example()
        for split in ('test', 'val'):
            for key, evidence in [('source', source), ('witness', witness), ('guard', guard)]:
                result = self.run_gate(**{key: replace(evidence, frame=replace(evidence.frame, split=split))})
                self.assertFalse(result.accepted)
                self.assertIn('cannot control optimization', result.reason)

    def test_rejects_same_view_and_dynamic_unsynchronized_view(self):
        source, witness, _, _ = example()
        same_pose = replace(witness, frame=replace(witness.frame, viewpoint_id=source.frame.viewpoint_id))
        self.assertFalse(self.run_gate(witness=same_pose).accepted)
        wrong_time = replace(witness, frame=replace(witness.frame, timestamp=.1))
        self.assertFalse(self.run_gate(witness=wrong_time, static_scene=False).accepted)
        self.assertTrue(self.run_gate(witness=wrong_time, static_scene=True).accepted)

    def test_rejects_empty_mask_invalid_calibration_and_nan(self):
        source, witness, guard, _ = example()
        invalid = [replace(witness, weight=plane(0)),
                   replace(witness, geometry=plane(float('nan'))),
                   replace(witness, frame=replace(witness.frame, calibrated=False))]
        for item in invalid:
            self.assertFalse(self.run_gate(witness=item).accepted)
        self.assertFalse(self.run_gate(guard=replace(guard, weight=plane(0))).accepted)
        self.assertFalse(self.run_gate(source=replace(source, noise=plane(0))).accepted)

    def test_rejects_no_witness_or_missing_anchor(self):
        source, witness, guard, _ = example()
        self.assertFalse(decide_geometry_transfer(source, [], [guard]).accepted)
        self.assertFalse(decide_geometry_transfer(source, [witness], [guard]).accepted)

    def test_rejects_unbounded_sensor_change(self):
        result = self.run_gate(config=GateConfig(max_source_change=.1))
        self.assertFalse(result.accepted)
        self.assertIn('trust budget', result.reason)

    def test_huge_finite_residual_cannot_accept_no_improvement(self):
        source, witness, _, _ = example()
        huge = plane(1e20).float()
        source = replace(source, before=huge, geometry=huge, appearance=huge)
        witness = replace(witness, before=huge, geometry=huge, appearance=huge)
        self.assertFalse(self.run_gate(source=source, witness=witness).accepted)

    def test_residual_division_overflow_fails_closed(self):
        source, _, _, _ = example()
        source = replace(source, before=plane(1e30).float(), noise=plane(1e-30).float())
        self.assertFalse(self.run_gate(source=source).accepted)


if __name__ == '__main__':
    unittest.main()
