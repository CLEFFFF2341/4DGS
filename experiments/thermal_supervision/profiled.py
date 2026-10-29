"""V2: source appearance profiling followed by independently witnessed geometry.

An experimental fixed-topology adapter, not a claim of causal identification.
The thermal head has no persistent optimizer: accepted head and delta form one
transaction. Witnesses only veto the single source-selected candidate.
"""
from dataclasses import asdict, dataclass
import math
import torch

from .core import sensor_loss, observation_change
from .engine import TransferEngine, SearchConfig, normalized_proposal


@dataclass(frozen=True)
class ProfileConfig(SearchConfig):
    appearance_steps: int = 8
    min_baseline_scales: float = .5
    min_local_pixels: int = 64
    min_local_coverage: float = .5
    support_threshold: float = .01


class ProfiledTransferEngine(TransferEngine):
    def __init__(self, model, config=ProfileConfig()):
        super().__init__(model, config)
        if config.appearance_steps < 1 or config.min_local_pixels < 1:
            raise ValueError('positive profiling budget and support required')
        if not 0 < config.min_local_coverage <= 1 or not 0 < config.support_threshold < 1:
            raise ValueError('invalid local support thresholds')
        if not math.isfinite(config.min_baseline_scales) or config.min_baseline_scales <= 0:
            raise ValueError('positive physical baseline required')

    def _preflight(self, source, witnesses, guards):
        super()._preflight(source, witnesses, guards)
        for obs in [source, *witnesses]:
            if int((obs.weight > 0).sum()) < self.config.min_local_pixels:
                raise ValueError('insufficient thermal support')
        def center(o):
            return torch.linalg.inv(o.samples[len(o.samples)//2].camera.w2c)[:3, 3]
        baseline = float(self.model.reference_scale.median()) * self.config.min_baseline_scales
        for witness in witnesses:
            if float((center(source)-center(witness)).norm()) < baseline:
                raise ValueError('thermal witness physical baseline too small')

    def step(self, source, witnesses, guards):
        if self.config.thermal_weight == 0:
            return {'accepted': False, 'reason': 'thermal disabled', 'renders': 0}
        self._preflight(source, witnesses, guards)
        m, c = self.model, self.config
        version, calls = m.version, m.render_calls
        record = {'accepted': False, 'reason': '', 'start_version': version}
        def loss(o, prediction):
            return sensor_loss(prediction, o.target, o.weight, o.noise)
        def finish(reason):
            record.update(reason=reason, end_version=m.version, renders=m.render_calls-calls)
            self.transactions += 1
            return record
        # Fixed-budget nuisance profiling. All line searches use source only.
        head = m.thermal.detach().clone()
        original_source = m.render(source).detach()
        original_loss = float(loss(source, original_source))
        for _ in range(c.appearance_steps):
            leaf = head.detach().requires_grad_()
            value = loss(source, m.render(source, thermal=leaf))
            grad, = torch.autograd.grad(value, (leaf,))
            best_loss, best = float(value.detach()), head
            with torch.no_grad():
                for radius in c.appearance_radii:
                    candidate = normalized_proposal(leaf, grad, torch.ones_like(leaf), radius)
                    image = m.render(source, thermal=candidate)
                    score = float(loss(source, image))
                    change = float(observation_change(original_source, image, source.weight, source.noise))
                    if score < best_loss and change <= c.gate.max_source_change:
                        best_loss, best = score, candidate
            head = best.detach()
        delta = m.thermal_delta.detach().clone().requires_grad_()
        profiled_source = m.render(source, delta=delta, thermal=head)
        base = loss(source, profiled_source)
        gradient, = torch.autograd.grad(base, (delta,))
        record.update(source_original_loss=original_loss, source_profiled_loss=float(base.detach()),
                      geometry_grad_norm=float(gradient.norm()))
        candidates = []
        with torch.no_grad():
            for radius in c.geometry_radii:
                candidate = normalized_proposal(delta, gradient, m.reference_scale, radius*c.thermal_weight)
                if float((candidate/m.reference_scale).norm(dim=-1).max()) > c.cumulative_radius:
                    continue
                image = m.render(source, delta=candidate, thermal=head)
                gain = float(base) - float(loss(source, image))
                change = float(observation_change(original_source, image, source.weight, source.noise))
                if gain > c.gate.min_source_gain and change <= c.gate.max_source_change:
                    candidates.append((gain, candidate, radius))
            if not candidates:
                return finish('no source residual geometry improvement')
            gain, candidate, radius = max(candidates, key=lambda item: item[0])
            record.update(source_geometry_gain=gain, geometry_radius=radius)
            # No more source candidate search after any witness is rendered.
            witness_gains = []
            for obs in witnesses:
                live = float(loss(obs, m.render(obs)))
                appearance = float(loss(obs, m.render(obs, thermal=head)))
                proposed = float(loss(obs, m.render(obs, delta=candidate, thermal=head)))
                witness_gains.append({'over_profiled': appearance-proposed, 'over_live': live-proposed})
                record['witness_gains'] = witness_gains
                if appearance-proposed <= c.gate.appearance_advantage_margin or live-proposed <= c.gate.min_witness_gain:
                    return finish('independent witness rejects profiled geometry')
            changed = (candidate-m.thermal_delta).norm(dim=-1) > 1e-8
            coverage = []
            for obs in guards:
                before, after = m.render(obs), m.render(obs, delta=candidate)
                # Render selected primitives through true alpha compositing;
                # support covers both the old and proposed projections.
                support = torch.maximum(m.render(obs, support=changed).mean(0, keepdim=True),
                    m.render(obs, delta=candidate, support=changed).mean(0, keepdim=True)) > c.support_threshold
                reliable = obs.weight > 0
                local = support & reliable
                fraction = float(local.sum()) / max(1, int(support.sum()))
                coverage.append(fraction)
                record['rgb_local_coverage'] = coverage
                if int(local.sum()) < c.min_local_pixels or fraction < c.min_local_coverage:
                    return finish('insufficient RGB support on changed geometry')
                before_loss, after_loss = float(loss(obs, before)), float(loss(obs, after))
                anchor = self.anchors.setdefault(obs.frame.view_id, before_loss)
                local_weight = obs.weight * local
                local_change = float(sensor_loss(after, obs.target, local_weight, obs.noise) -
                                     sensor_loss(before, obs.target, local_weight, obs.noise))
                if after_loss > before_loss+c.gate.max_rgb_loss_increase or after_loss > anchor+c.gate.max_rgb_anchor_increase or local_change > c.gate.max_rgb_loss_increase:
                    return finish('RGB local/global/anchor guard rejects geometry')
            if m.version != version:
                raise RuntimeError('model changed during transaction')
            # Exact witnessed pair is committed. No stale thermal Adam moments.
            m.thermal_delta.copy_(candidate)
            m.thermal.copy_(head)
            m.version += 1
            record['accepted'] = True
            return finish('profiled geometry accepted')
