"""CPU invariants for residual/optimizer correspondence under native topology ops."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import unittest
import torch
from .model import ResidualGaussians


class ResidualTests(unittest.TestCase):
    def test_prune_sort_and_update(self):
        g = ResidualGaussians(3, False)
        shapes = {'spatial_means':(7,3), 'temporal_means':(7,1),
            'spatial_scales':(7,3), 'temporal_scales':(7,1),
            'left_isoclinic_rotations':(7,4), 'right_isoclinic_rotations':(7,4),
            'opacities':(7,1), 'sh_coefficients_0':(7,1,3), 'sh_coefficients_rest':(7,15,3)}
        groups = []
        for key, shape in shapes.items():
            value = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)+1
            param = torch.nn.Parameter(value)
            setattr(g, '_'+key, param)
            groups.append({'name':key, 'params':[param]})
        g.optimizer = torch.optim.Adam(groups, lr=.01)
        g.point_groups = ['accel','color']
        for key, shape in [('accel',(7,3)),('color',(7,1,3)),('affine',(8,3,4))]:
            g.residuals[key] = torch.nn.Parameter(torch.arange(torch.tensor(shape).prod().item()).float().reshape(shape)+2)
        g.extra_optimizer = torch.optim.Adam([{'name':k,'params':[v]} for k,v in g.residuals.items()], lr=.01)
        g.reference_sigma_t = torch.arange(7).float()[:,None]+1
        g.reference_sigma_xyz = torch.arange(21).float().reshape(7,3)+1
        sum(p.square().sum() for p in g.parameters()).backward()
        g.optimizer.step(); g.extra_optimizer.step()
        g.optimizer.zero_grad(); g.extra_optimizer.zero_grad()
        old = {k:v.detach().clone() for k,v in g.residuals.items()}
        moments = {group['name']: g.extra_optimizer.state[group['params'][0]]['exp_avg'].clone()
                   for group in g.extra_optimizer.param_groups}
        old_sigma = g.reference_sigma_xyz.clone()
        mask = torch.tensor([False,True,False,False,True,False,False])
        order = torch.tensor([4,1,0,3,2])
        g.prune(mask); g.sort(order)
        for group in g.extra_optimizer.param_groups:
            key, param = group['name'], group['params'][0]
            self.assertIs(g.residuals[key], param)
            expected = old[key][~mask][order] if key in g.point_groups else old[key]
            expected_moment = moments[key][~mask][order] if key in g.point_groups else moments[key]
            torch.testing.assert_close(param, expected)
            torch.testing.assert_close(g.extra_optimizer.state[param]['exp_avg'], expected_moment)
        torch.testing.assert_close(g.reference_sigma_xyz, old_sigma[~mask][order])
        previous = g.residuals['accel'].detach().clone()
        g.residuals['accel'].square().sum().backward()
        g.extra_optimizer.step()
        self.assertFalse(torch.equal(previous, g.residuals['accel']))

    def test_acceleration_preserves_center_and_tangent(self):
        timestamp = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
        tau, sigma = .4, .2
        delta = .03 * 1.7 * ((timestamp-tau)/sigma).clamp(-3,3).square()
        first = torch.autograd.grad(delta, timestamp, create_graph=True)[0]
        second = torch.autograd.grad(first, timestamp)[0]
        self.assertEqual(float(delta), 0.)
        self.assertEqual(float(first), 0.)
        self.assertGreater(float(second), 0.)


if __name__ == '__main__':
    unittest.main()
