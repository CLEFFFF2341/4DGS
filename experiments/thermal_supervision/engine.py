"""Source-only finite search and atomic thermal geometry transactions."""
from dataclasses import asdict, dataclass, field
import math
from typing import Sequence

import torch

from .core import (GateConfig, GateDecision, ThermalEvidence, RGBEvidence,
                   bounded_proposal, decide_geometry_transfer, sensor_loss, observation_change)
from .runtime import ThermalModel, Observation


@dataclass(frozen=True)
class SearchConfig:
    geometry_radii: tuple[float, ...] = (.5, .25, .125, .0625)
    appearance_radii: tuple[float, ...] = (1., .5, .25, .125, .0625, .03125)
    cumulative_radius: float = 2.
    thermal_weight: float = 1.
    static_scene: bool = True
    gate: GateConfig = field(default_factory=GateConfig)


def normalized_proposal(value, gradient, scale, radius):
    # Radius is a displacement budget in reference-scale units, independent
    # of image pixel count / mean-loss normalization. No Adam momentum.
    direction_norm = (gradient.detach() * scale).norm(dim=-1, keepdim=True)
    return bounded_proposal(value, gradient / direction_norm.clamp_min(1e-12), scale, radius)


class TransferEngine:
    def __init__(self, model: ThermalModel, config=SearchConfig()):
        self.model, self.config = model, config
        if not math.isfinite(config.thermal_weight) or config.thermal_weight < 0:
            raise ValueError('thermal_weight must be finite and nonnegative')
        if not math.isfinite(config.cumulative_radius) or config.cumulative_radius <= 0:
            raise ValueError('invalid cumulative radius')
        if any(not math.isfinite(x) or x <= 0 for x in (*config.geometry_radii, *config.appearance_radii)):
            raise ValueError('invalid source search radii')
        gate = config.gate
        thresholds = (gate.min_source_gain, gate.min_witness_gain, gate.appearance_advantage_margin,
                      gate.max_rgb_loss_increase, gate.max_rgb_anchor_increase, gate.synchronization_tolerance)
        if (any(not math.isfinite(x) or x < 0 for x in thresholds) or gate.min_pixels < 1
                or gate.min_witness_views < 1 or not 0 <= gate.source_gain_relative_tolerance < 1
                or not math.isfinite(gate.max_source_change) or gate.max_source_change <= 0):
            raise ValueError('invalid gate configuration')
        self.anchors = {}
        self.transactions = 0

    def state_dict(self):
        return {'anchors': dict(self.anchors), 'transactions': self.transactions,
                'config': asdict(self.config)}

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('resumed search configuration differs')
        if any(not math.isfinite(v) or v < 0 for v in state['anchors'].values()):
            raise ValueError('invalid restored RGB anchor')
        if not isinstance(state['transactions'], int) or state['transactions'] < 0:
            raise ValueError('invalid restored transaction count')
        self.anchors = dict(state['anchors'])
        self.transactions = int(state['transactions'])

    def _preflight(self, source, witnesses, guards):
        observations = [source, *witnesses, *guards]
        if source.sensor != 'thermal' or any(o.sensor != 'thermal' for o in witnesses) or any(o.sensor != 'rgb' for o in guards):
            raise ValueError('source/witness must be thermal, guard must be RGB')
        if len(witnesses) < self.config.gate.min_witness_views or not guards:
            raise ValueError('independent thermal witness and RGB guard required')
        for obs in observations:
            if obs.frame.split != 'train' or not obs.frame.calibrated:
                raise ValueError('only audited TRAIN observations may control updates')
            if not math.isfinite(obs.frame.timestamp):
                raise ValueError('finite physical timestamp required')
            obs.validate()
            if int((obs.weight > 0).sum()) < self.config.gate.min_pixels:
                raise ValueError('insufficient fixed observation support')
        thermal = [source, *witnesses]
        if len({o.frame.view_id for o in thermal}) != len(thermal) or len({o.frame.viewpoint_id for o in thermal}) != len(thermal):
            raise ValueError('thermal witnesses must be distinct viewpoints')
        if not self.config.static_scene and any(abs(o.frame.timestamp-source.frame.timestamp) > self.config.gate.synchronization_tolerance for o in observations):
            raise ValueError('dynamic evidence is not synchronized')

    def step(self, source: Observation, witnesses: Sequence[Observation], guards: Sequence[Observation]):
        # Disabled means literally no input read/render/state/RNG change.
        if self.config.thermal_weight == 0:
            return {'accepted': False, 'reason': 'thermal disabled', 'renders': 0}
        self._preflight(source, witnesses, guards)
        model, config = self.model, self.config
        version, start_calls = model.version, model.render_calls
        delta = model.thermal_delta.detach().clone().requires_grad_()
        head = model.thermal.detach().clone().requires_grad_()
        before = model.render(source, delta=delta, thermal=head)
        loss = sensor_loss(before, source.target, source.weight, source.noise)
        grad_g, grad_a = torch.autograd.grad(loss, (delta, head))
        source_before = before.detach()
        base_loss = float(loss.detach())
        # Search does not access witnesses/guards. Select G then match A on source.
        geometry = []
        appearance = []
        with torch.no_grad():
            for radius in config.geometry_radii:
                proposal = normalized_proposal(delta, grad_g, model.reference_scale, radius * config.thermal_weight)
                if float((proposal / model.reference_scale).norm(dim=-1).max()) > config.cumulative_radius:
                    continue
                image = model.render(source, delta=proposal)
                change = float(observation_change(source_before, image, source.weight, source.noise))
                gain = base_loss - float(sensor_loss(image, source.target, source.weight, source.noise))
                if math.isfinite(gain) and change <= config.gate.max_source_change and gain > config.gate.min_source_gain:
                    geometry.append((gain, proposal, image, radius))
            for radius in config.appearance_radii:
                proposal = normalized_proposal(head, grad_a, torch.ones_like(head), radius)
                image = model.render(source, thermal=proposal)
                change = float(observation_change(source_before, image, source.weight, source.noise))
                gain = base_loss - float(sensor_loss(image, source.target, source.weight, source.noise))
                if math.isfinite(gain) and change <= config.gate.max_source_change:
                    appearance.append((gain, proposal, image, radius))
        record = {'accepted': False, 'reason': '', 'source_base_loss': base_loss,
                  'geometry_grad_norm': float(grad_g.norm()), 'appearance_grad_norm': float(grad_a.norm()),
                  'geometry_source_gains': [p[0] for p in geometry],
                  'appearance_source_gains': [p[0] for p in appearance],
                  'start_version': version}
        if not geometry or not appearance:
            record['reason'] = 'no finite improving geometry / bounded appearance candidate'
        else:
            g = max(geometry, key=lambda p: p[0])
            a = min(appearance, key=lambda p: abs(p[0] - g[0]))
            record.update(source_geometry_gain=g[0], source_appearance_gain=a[0],
                          candidate_geometry_gain=g[0], candidate_appearance_gain=a[0],
                          geometry_radius=g[3], appearance_radius=a[3])
            with torch.no_grad():
                se = ThermalEvidence(source.frame, source.target, source_before, g[2], a[2], source.weight, source.noise)
                we = [ThermalEvidence(o.frame, o.target, model.render(o), model.render(o, delta=g[1]),
                      model.render(o, thermal=a[1]), o.weight, o.noise) for o in witnesses]
                re = []
                anchors = []
                for obs in guards:
                    image = model.render(obs)
                    anchor = self.anchors.get(obs.frame.view_id)
                    if anchor is None:
                        anchor = float(sensor_loss(image, obs.target, obs.weight, obs.noise))
                        self.anchors[obs.frame.view_id] = anchor
                    anchors.append(anchor)
                    re.append(RGBEvidence(obs.frame, obs.target, image, model.render(obs, delta=g[1]), obs.weight, obs.noise))
                decision = decide_geometry_transfer(se, we, re, config=config.gate,
                    static_scene=config.static_scene, rgb_anchor_losses=anchors)
                record.update(asdict(decision))
                if model.version != version:
                    raise RuntimeError('model changed during thermal transaction')
                if decision.accepted:
                    model.thermal_delta.copy_(g[1])
                    model.version += 1
        self.transactions += 1
        record['renders'] = model.render_calls-start_calls
        record['end_version'] = model.version
        return record
