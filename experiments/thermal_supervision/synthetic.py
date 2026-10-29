"""Known-geometry CUDA fixtures; never presented as independent thermal data."""
from dataclasses import replace
import math

import torch

from .core import Frame
from .runtime import Camera, ExposureSample, Observation, ThermalModel


def make_model(seed=0, device='cuda'):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    xy = torch.cartesian_prod(torch.linspace(-.65, .65, 5), torch.linspace(-.45, .45, 4))
    n = len(xy)
    xyz = torch.cat((xy, 2.6 + .2 * torch.rand((n, 1), generator=generator)), 1)
    color = .15 + .7 * torch.rand((n, 3), generator=generator)
    q = torch.zeros(n, 4)
    q[:, 0] = 1
    raw = {'spatial_means': xyz, 'temporal_means': torch.ones(n, 1) * .5,
           'spatial_scales': torch.full((n, 3), math.log(.11)),
           'temporal_scales': torch.full((n, 1), math.log(2.)),
           'left_isoclinic_rotations': q.clone(), 'right_isoclinic_rotations': q.clone(),
           'opacities': torch.full((n, 1), 2.),
           'sh_coefficients_0': ((color-.5)/.28209479177387814)[:, None, :],
           'sh_coefficients_rest': torch.zeros(n, 15, 3)}
    model = ThermalModel({k: v.to(device) for k, v in raw.items()})
    heat = .25 + .7 * torch.rand((n, 1), generator=generator)
    with torch.no_grad():
        model.thermal.copy_(torch.log(torch.expm1(heat.to(device))))
    return model


def clone_model(model):
    result = ThermalModel(model.raw, active_sh_bases=model.active_sh_bases,
                          thermal_degree=model.thermal_degree, reference_scale=model.reference_scale)
    result.load_state_dict(model.state_dict())
    return result


def observation(model, view_id, x, sensor='thermal', *, split='train', blurred=False, gain=1., timestamp=.5):
    device = model.thermal.device
    width, height, focal = (64, 48, 65.) if sensor == 'thermal' else (80, 60, 81.25)
    def camera(dx):
        matrix = torch.eye(4, device=device)
        matrix[0, 3] = -(x + dx + (.025 if sensor == 'thermal' else 0))
        return Camera(matrix, width, height, focal, focal, width/2, height/2)
    offsets = (-.09, 0., .09) if blurred and sensor == 'rgb' else (0.,)
    samples = tuple(ExposureSample(camera(dx), timestamp, 1/len(offsets)) for dx in offsets)
    shape = (3 if sensor == 'rgb' else 1, height, width)
    obs = Observation(Frame(view_id+'_'+sensor, view_id, timestamp, split, True), sensor, samples,
                      torch.zeros(shape, device=device), torch.ones((1, height, width), device=device),
                      torch.tensor(.04 if sensor == 'thermal' else .02, device=device), gain)
    with torch.no_grad():
        return replace(obs, target=model.render(obs).detach())


def make_case(seed=0, case='geometry', device='cuda'):
    truth = make_model(seed, device)
    model = clone_model(truth)
    with torch.no_grad():
        if case in ('geometry', 'mixed', 'cold_start'):
            model.raw['spatial_means'][:, 0].add_(.055)
        if case in ('appearance', 'mixed'):
            model.thermal.add_(.35)
        if case == 'cold_start':
            model.thermal.fill_(math.log(math.expm1(.5)))
        if case not in ('geometry', 'mixed', 'appearance', 'unchanged', 'cold_start', 'thermal_only_shift'):
            raise ValueError(case)
    heat_truth = truth
    if case == 'thermal_only_shift':
        heat_truth = clone_model(truth)
        with torch.no_grad():
            heat_truth.raw['spatial_means'][:, 0].add_(.055)
    source = observation(heat_truth, 'source', -.35)
    witness = observation(heat_truth, 'witness', .35)
    guard = observation(truth, 'witness', .35, 'rgb', blurred=True, gain=.25)
    test = observation(truth, 'unseen', .12, 'rgb', split='test', gain=.25)
    return truth, model, source, witness, guard, test
