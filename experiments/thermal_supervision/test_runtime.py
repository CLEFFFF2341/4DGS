"""Independent runtime contract checks; tiny native renders, not a benchmark."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from .core import Frame, GateConfig, sensor_loss
from .engine import SearchConfig, TransferEngine
from .profiled import ProfileConfig, ProfiledTransferEngine
from .runtime import Camera, ExposureSample, Observation, ThermalModel
from .synthetic import make_model, make_case, observation


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


class RuntimeCPUContracts(unittest.TestCase):
    def test_both_engines_validate_guard_thresholds(self):
        model = make_model(device='cpu')
        for engine_type, config_type in ((TransferEngine, SearchConfig),
                                         (ProfiledTransferEngine, ProfileConfig)):
            for gate in (GateConfig(max_rgb_loss_increase=float('nan')),
                         GateConfig(max_rgb_anchor_increase=float('inf')),
                         GateConfig(appearance_advantage_margin=-1.)):
                with self.subTest(engine=engine_type.__name__, gate=gate):
                    with self.assertRaises(ValueError):
                        engine_type(model, config_type(gate=gate))

    def test_camera_rejects_nonrigid_and_reflection(self):
        for matrix in (torch.diag(torch.tensor([2., 1., 1., 1.])),
                       torch.diag(torch.tensor([-1., 1., 1., 1.]))):
            with self.assertRaises((ValueError, AssertionError)):
                Camera(matrix, 32, 24, 30., 30., 16., 12.).validate()

    def test_observation_rejects_noise_rank_and_empty_exposure(self):
        camera = Camera(torch.eye(4), 8, 8, 10., 10., 4., 4.)
        obs = Observation(Frame('s', 's', 0., calibrated=True), 'thermal',
                          (ExposureSample(camera, 0., 1.),), torch.zeros(1, 8, 8),
                          torch.ones(1, 8, 8), torch.ones(2, 1, 1, 1))
        with self.assertRaises((ValueError, RuntimeError)):
            obs.validate()
        with self.assertRaises(ValueError):
            replace(obs, samples=(), noise=torch.tensor(1.)).validate()

    def test_disabled_engine_does_not_even_read_evidence(self):
        model = make_model(device='cpu')
        before, rng = snapshot(model), torch.get_rng_state().clone()
        engine = TransferEngine(model, SearchConfig(thermal_weight=0.))
        result = engine.step(None, None, None)
        self.assertFalse(result['accepted'])
        self.assertEqual(model.version, 0)
        self.assertEqual(model.render_calls, 0)
        self.assertEqual(engine.transactions, 0)
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_constructor_rejects_cast_overflow(self):
        model = make_model(device='cpu')
        raw = {key: value.detach().double().clone() for key, value in model.raw.items()}
        raw['spatial_means'][0, 0] = 1e100
        with self.assertRaises(ValueError):
            ThermalModel(raw)
        with self.assertRaises(ValueError):
            ThermalModel(model.raw, reference_scale=torch.full((20, 3), 1e100, dtype=torch.float64))

    def test_reindex_preserves_adam_rows_and_buffers(self):
        model = make_model(device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for parameter in model.parameters():
            parameter.grad = torch.arange(1, len(parameter)+1, dtype=parameter.dtype).reshape(
                (-1,) + (1,) * (parameter.ndim-1)).expand_as(parameter).clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        values = {key: p.detach().clone() for key, p in parameters.items()}
        moments = {key: optimizer.state[p]['exp_avg'].clone() for key, p in parameters.items()}
        buffers = {key: value.clone() for key, value in model.named_buffers()}
        indices = torch.tensor([19, 2, 7, 0])
        model.reindex(indices, [optimizer])
        current = dict(model.named_parameters())
        for key, value in values.items():
            torch.testing.assert_close(current[key], value[indices])
            torch.testing.assert_close(optimizer.state[current[key]]['exp_avg'], moments[key][indices])
        for key, value in buffers.items():
            torch.testing.assert_close(dict(model.named_buffers())[key], value[indices])
        self.assertEqual(model.version, 1)
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']},
                         {id(p) for p in model.parameters()})

    def test_invalid_reindex_is_atomic(self):
        model = make_model(device='cpu')
        before = snapshot(model)
        old_ids = {key: id(value) for key, value in model.named_parameters()}
        foreign = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([*model.parameters(), foreign])
        with self.assertRaises(ValueError):
            model.reindex(torch.tensor([2, 0, 1]), [optimizer])
        self.assertEqual(old_ids, {key: id(value) for key, value in model.named_parameters()})
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_duplicate_reindex_rejected_without_mutation(self):
        model = make_model(device='cpu')
        before = snapshot(model)
        with self.assertRaises(ValueError):
            model.reindex(torch.tensor([0, 0]))
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_checkpoint_roundtrip_preserves_residual_identity_and_engine_state(self):
        model = make_model(device='cpu')
        model.thermal_delta[0] = torch.tensor([.01, -.02, .03])
        model.version = 7
        engine = TransferEngine(model)
        engine.anchors['guard_rgb'] = .125
        engine.transactions = 5
        optimizer = torch.optim.Adam(model.raw.parameters(), lr=.002)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pt'
            model.save(path, optimizers={'rgb': optimizer}, extra={'engine': engine.state_dict()})
            loaded, payload = ThermalModel.load(path, device='cpu')
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(loaded.version, 7)
        resumed = TransferEngine(loaded)
        resumed.load_state_dict(payload['extra']['engine'])
        self.assertEqual(resumed.state_dict(), engine.state_dict())
        self.assertIn('rgb', payload['optimizers'])


@unittest.skipUnless(torch.cuda.is_available(), 'native CUDA runtime unavailable')
class RuntimeCUDAContracts(unittest.TestCase):
    def test_real_thermal_gradients_leave_original_parameters_untouched(self):
        _, model, source, _, _, _ = make_case(case='mixed')
        before = snapshot(model)
        delta = model.thermal_delta.detach().clone().requires_grad_()
        head = model.thermal.detach().clone().requires_grad_()
        prediction = model.render(source, delta=delta, thermal=head)
        sensor_loss(prediction, source.target, source.weight, source.noise).backward()
        for value in (delta.grad, head.grad):
            self.assertIsNotNone(value)
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertGreater(float(value.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_native_geometry_and_thermal_gradients_match_finite_difference(self):
        _, model, source, _, _, _ = make_case(case='mixed')
        for kind in ('delta', 'thermal'):
            leaf = (model.thermal_delta if kind == 'delta' else model.thermal).detach().clone().requires_grad_()
            prediction = model.render(source, **{kind: leaf})
            loss = sensor_loss(prediction, source.target, source.weight, source.noise)
            gradient, = torch.autograd.grad(loss, leaf)
            flat_index = int(gradient.abs().argmax())
            analytic = float(gradient.view(-1)[flat_index])
            self.assertGreater(abs(analytic), 1e-4)
            # Full-image geometric FD at 1e-4 crosses an alpha=1/255 raster
            # cutoff in this fixture. Test two finite scales, plus the fixed
            # smooth interior separately below; do not claim everywhere-C1.
            for eps in (1e-3, 3e-3):
                plus, minus = leaf.detach().clone(), leaf.detach().clone()
                plus.view(-1)[flat_index] += eps
                minus.view(-1)[flat_index] -= eps
                with torch.no_grad():
                    lp = sensor_loss(model.render(source, **{kind: plus}), source.target, source.weight, source.noise)
                    lm = sensor_loss(model.render(source, **{kind: minus}), source.target, source.weight, source.noise)
                numerical = float((lp-lm)/(2*eps))
                tolerance = .05 if kind == 'delta' else .01
                self.assertAlmostEqual(analytic, numerical, delta=max(2e-3, tolerance*abs(analytic)), msg=f'{kind}, eps={eps}')

    def test_geometry_gradient_on_fixed_interior_at_small_step(self):
        _, model, source, _, _, _ = make_case(case='mixed')
        leaf = model.thermal_delta.detach().clone().requires_grad_()
        loss = sensor_loss(model.render(source, delta=leaf), source.target, source.weight, source.noise)
        gradient, = torch.autograd.grad(loss, leaf)
        index = int(gradient.abs().argmax())
        selected = torch.zeros(len(leaf), device=leaf.device, dtype=torch.bool)
        selected[index//3] = True
        with torch.no_grad():
            support = model.render(source, support=selected)
            weight = source.weight * (support > .05)
        self.assertGreater(int(weight.sum()), 64)
        value = sensor_loss(model.render(source, delta=leaf), source.target, weight, source.noise)
        interior_gradient, = torch.autograd.grad(value, leaf)
        eps = 1e-4
        plus, minus = leaf.detach().clone(), leaf.detach().clone()
        plus.view(-1)[index] += eps
        minus.view(-1)[index] -= eps
        with torch.no_grad():
            lp = sensor_loss(model.render(source, delta=plus), source.target, weight, source.noise)
            lm = sensor_loss(model.render(source, delta=minus), source.target, weight, source.noise)
        analytic = float(interior_gradient.view(-1)[index])
        self.assertAlmostEqual(analytic, float((lp-lm)/(2*eps)), delta=.01*abs(analytic))

    def test_rgb_updates_have_no_thermal_gradient(self):
        _, model, _, _, guard, _ = make_case(case='geometry')
        loss = sensor_loss(model.render(guard, train_raw=True), guard.target, guard.weight, guard.noise)
        loss.backward()
        self.assertIsNone(model.thermal.grad)
        self.assertTrue(bool(torch.isfinite(model.raw['spatial_means'].grad).all()))
        self.assertGreater(float(model.raw['spatial_means'].grad.norm()), 0.)
        self.assertIsNone(model.thermal_delta.grad)

    def test_exposure_matches_individual_native_renders(self):
        truth = make_model()
        obs = observation(truth, 'blur', .1, 'rgb', blurred=True, gain=.25)
        with torch.no_grad():
            integrated = truth.render(obs)
            individual = [truth.render(replace(obs, samples=(sample,))) for sample in obs.samples]
        expected = sum(image*sample.weight for image, sample in zip(individual, obs.samples)) / sum(s.weight for s in obs.samples)
        torch.testing.assert_close(integrated, expected, rtol=1e-5, atol=1e-7)
        self.assertGreater(float((individual[0]-individual[-1]).abs().max()), 1e-3)

    def test_rejected_transaction_preserves_model_gradients_and_rng(self):
        _, model, source, witness, guard, _ = make_case(case='unchanged')
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        before = snapshot(model)
        gradients = {key: p.grad.clone() for key, p in model.named_parameters()}
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        result = TransferEngine(model).step(source, [witness], [guard])
        self.assertFalse(result['accepted'])
        self.assertEqual(model.version, 0)
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for key, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), gpu_rng, rtol=0, atol=0)

    def test_test_split_is_rejected_before_any_render(self):
        _, model, source, witness, guard, test = make_case()
        start = model.render_calls
        with self.assertRaises(ValueError):
            TransferEngine(model).step(source, [witness], [test])
        self.assertEqual(model.render_calls, start)

    def test_profiled_requires_configured_witness_count_and_finite_time(self):
        _, model, source, witness, guard, _ = make_case()
        calls = model.render_calls
        with self.assertRaises(ValueError):
            ProfiledTransferEngine(model, ProfileConfig(gate=GateConfig(min_witness_views=2))).step(
                source, [witness], [guard])
        invalid = replace(source, frame=replace(source.frame, timestamp=float('nan')))
        with self.assertRaises(ValueError):
            ProfiledTransferEngine(model).step(invalid, [witness], [guard])
        self.assertEqual(model.render_calls, calls)

    def test_profiled_source_search_finishes_before_witness_render(self):
        _, model, source, witness, guard, _ = make_case()
        order = []
        render = model.render
        def traced(observation, **kwargs):
            order.append(observation.frame.view_id)
            return render(observation, **kwargs)
        model.render = traced
        ProfiledTransferEngine(model).step(source, [witness], [guard])
        witness_index = order.index(witness.frame.view_id)
        self.assertTrue(all(view == source.frame.view_id for view in order[:witness_index]))
        self.assertNotIn(source.frame.view_id, order[witness_index:])

    def test_checkpoint_roundtrip_preserves_actual_render(self):
        _, model, _, _, _, test = make_case()
        model.thermal_delta[0, 0] = -.02
        with torch.no_grad():
            before = model.render(test)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'native.pt'
            model.save(path)
            loaded, _ = ThermalModel.load(path, device='cuda')
        with torch.no_grad():
            after = loaded.render(test)
        torch.testing.assert_close(after, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
