import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .core import GateConfig
from .data import load_manifest
from .engine import SearchConfig, TransferEngine
from .profiled import ProfileConfig, ProfiledTransferEngine
from .runtime import ThermalModel
from .synthetic import make_case


class ManifestTests(unittest.TestCase):
    def test_restored_anchor_nan_rejected(self):
        from .synthetic import make_model
        engine = ProfiledTransferEngine(make_model(device='cpu'))
        state = engine.state_dict()
        state['anchors'] = {'guard': float('nan')}
        with self.assertRaisesRegex(ValueError, 'anchor'):
            engine.load_state_dict(state)

    def test_heldout_rejected_before_file_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps({'schema': 1, 'units': 'linear_sensor', 'static_scene': True,
                'observations': [{'id': 'heldout', 'split': 'test', 'calibrated': True}],
                'transactions': [{'source': 'heldout', 'witnesses': [], 'guards': []}]}))
            with self.assertRaisesRegex(ValueError, 'training'):
                load_manifest(path, 'cpu')


@unittest.skipUnless(torch.cuda.is_available(), 'native CUDA required')
class ProfiledTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def engine(self, model):
        return ProfiledTransferEngine(model, ProfileConfig(gate=GateConfig(min_source_gain=1e-5,
            min_witness_gain=1e-5, appearance_advantage_margin=1e-5, max_source_change=2.,
            max_rgb_loss_increase=1e-6, max_rgb_anchor_increase=1e-6)))

    def test_positive_geometry_and_no_native_gradient(self):
        truth, m, source, witness, guard, test = make_case(0)
        before = float((m.render(test)-test.target).square().mean())
        original = {k: v.clone() for k, v in m.raw.items()}
        record = self.engine(m).step(source, [witness], [guard])
        self.assertTrue(record['accepted'], record)
        self.assertLess(float((m.render(test)-test.target).square().mean()), before)
        for k, v in m.raw.items():
            torch.testing.assert_close(v, original[k], rtol=0, atol=0)
            self.assertIsNone(v.grad)

    def test_appearance_rejection_preserves_model_and_rng(self):
        _, m, source, witness, guard, _ = make_case(0, 'appearance')
        state = copy.deepcopy(m.state_dict())
        rng, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        record = self.engine(m).step(source, [witness], [guard])
        self.assertFalse(record['accepted'])
        for k, value in m.state_dict().items():
            torch.testing.assert_close(value, state[k], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(gpu, torch.cuda.get_rng_state()))

    def test_zero_weight_does_not_read_inputs(self):
        _, m, *_ = make_case(0)
        engine = ProfiledTransferEngine(m, ProfileConfig(thermal_weight=0))
        calls = m.render_calls
        self.assertEqual(engine.step(None, None, None)['renders'], 0)
        self.assertEqual(calls, m.render_calls)
        self.assertEqual(engine.transactions, 0)

    def test_same_camera_different_id_rejected(self):
        _, m, source, witness, guard, _ = make_case(0)
        witness = replace(witness, samples=source.samples)
        calls = m.render_calls
        with self.assertRaisesRegex(ValueError, 'baseline'):
            self.engine(m).step(source, [witness], [guard])
        self.assertEqual(calls, m.render_calls)

    def test_disconnected_rgb_mask_rejected(self):
        _, m, source, witness, guard, _ = make_case(0)
        mask = torch.zeros_like(guard.weight)
        mask[:, :4, :] = 1
        guard = replace(guard, weight=mask)
        record = self.engine(m).step(source, [witness], [guard])
        self.assertFalse(record['accepted'])
        self.assertIn('support', record['reason'])
        self.assertEqual(m.version, 0)

    def test_test_split_rejected_before_render(self):
        _, m, source, witness, guard, test = make_case(0)
        calls = m.render_calls
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            self.engine(m).step(source, [witness], [test])
        self.assertEqual(calls, m.render_calls)

    def test_thermal_only_shift_cannot_damage_rgb(self):
        _, m, source, witness, guard, _ = make_case(0, 'thermal_only_shift')
        record = self.engine(m).step(source, [witness], [guard])
        self.assertFalse(record['accepted'])
        self.assertIn('RGB', record['reason'])
        self.assertEqual(m.version, 0)

    def test_resume_matches_uninterrupted_transaction(self):
        _, m, source, witness, guard, _ = make_case(0)
        engine = self.engine(m)
        engine.step(source, [witness], [guard])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            m.save(path, extra={'engine': engine.state_dict()})
            loaded, payload = ThermalModel.load(path)
            resumed = self.engine(loaded)
            resumed.load_state_dict(payload['extra']['engine'])
            a = engine.step(source, [witness], [guard])
            b = resumed.step(source, [witness], [guard])
            self.assertEqual(a, b)
            for key, value in m.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    def test_manifest_cli_native_checkpoint_and_resume_audit(self):
        from .run import main
        _, m, source, witness, guard, _ = make_case(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for obs in (source, witness, guard):
                row = {'id': obs.frame.view_id, 'viewpoint_id': obs.frame.viewpoint_id,
                    'timestamp_seconds': obs.frame.timestamp, 'split': 'train', 'calibrated': True,
                    'sensor': obs.sensor, 'noise': float(obs.noise), 'gain': obs.gain, 'bias': obs.bias,
                    'exposure': []}
                for field in ('target', 'weight'):
                    row[field] = obs.frame.view_id+'_'+field+'.npy'
                    np.save(root/row[field], getattr(obs, field).cpu().numpy())
                for s in obs.samples:
                    camera = {k: getattr(s.camera, k) for k in ('width','height','fx','fy','cx','cy','near','far')}
                    camera['w2c'] = s.camera.w2c.cpu().tolist()
                    row['exposure'].append({'camera': camera, 'model_time': s.time, 'weight': s.weight})
                rows.append(row)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'schema': 1, 'units': 'linear_sensor', 'static_scene': True,
                'observations': rows, 'transactions': [{'source': source.frame.view_id,
                    'witnesses': [witness.frame.view_id], 'guards': [guard.frame.view_id]}]}))
            checkpoint = root/'native.pt'
            torch.save({'raw': {k: v.detach() for k, v in m.raw.items()}, 'active_sh_degree': 0}, checkpoint)
            argv = ['run', '--checkpoint', str(checkpoint), '--manifest', str(manifest), '--out', str(root/'result')]
            with patch('sys.argv', [*argv, '--audit-only']):
                main()
            self.assertFalse((root/'result').exists())
            with patch('sys.argv', argv):
                main()
            self.assertTrue((root/'result/checkpoint.pt').exists())
            argv[2] = str(root/'result/checkpoint.pt')
            with patch('sys.argv', [*argv, '--audit-only']):
                main()


if __name__ == '__main__':
    unittest.main()
