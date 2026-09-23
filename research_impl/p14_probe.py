from __future__ import annotations

import json
import math
import time

import torch

from .adapter import load_reference
from .cache import build_k1, build_k4
from .config import ROOT
from .runner import gpu_lock


@torch.no_grad()
def main() -> None:
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    with gpu_lock():
        adapter, scene = load_reference(load_cameras=True)
        k1_meta = build_k1(adapter, scene, splits["c4"], samples["T24"], samples["statistics_resolution"], ROOT / "research_cache" / "cut_roasted_beef")
        k4_meta = build_k4(adapter, [0, 74, 149, 224, 299], ROOT / "research_cache" / "cut_roasted_beef")
        k1 = torch.load(k1_meta["path"], map_location="cpu")
        k4 = torch.load(k4_meta["path"], map_location="cpu")
        model = adapter.model
        nd = adapter.dynamic_count
        base_opacity = model.get_motion_opacity.detach().clamp_min(1e-12)
        envelope_min = torch.full((nd,), float("inf"), device="cuda")
        envelope_max = torch.full((nd,), float("-inf"), device="cuda")
        reference_q = model.get_dynamic_rotation_at_t(149)
        reference_q = torch.nn.functional.normalize(reference_q, dim=1)
        max_angle = torch.zeros(nd, device="cuda")
        mean_position = torch.zeros((nd, 3), dtype=torch.float64, device="cuda")
        started = time.perf_counter()
        for timestamp in range(300):
            envelope = (model.get_motion_opacity_at_t(timestamp, training=False) / base_opacity).squeeze(1)
            envelope_min = torch.minimum(envelope_min, envelope)
            envelope_max = torch.maximum(envelope_max, envelope)
            rotation = torch.nn.functional.normalize(model.get_dynamic_rotation_at_t(timestamp), dim=1)
            dot = torch.sum(rotation * reference_q, dim=1).abs().clamp(max=1.0)
            max_angle = torch.maximum(max_angle, 2.0 * torch.acos(dot))
            mean_position += model.get_dynamic_xyz_at_t(timestamp).double() / 300.0
        max_motion = torch.zeros(nd, device="cuda")
        for timestamp in range(300):
            distance = torch.linalg.vector_norm(model.get_dynamic_xyz_at_t(timestamp).double() - mean_position, dim=1).float()
            max_motion = torch.maximum(max_motion, distance)
        opacity_gate = (envelope_min >= 0.99) & ((envelope_max - envelope_min) <= 0.01)
        rotation_gate = max_angle <= math.radians(2.0)
        candidate = opacity_gate & rotation_gate

        key_positions = model._xyz_motion.double()
        key_times = (torch.arange(key_positions.shape[1], device="cuda", dtype=torch.float64) * model.interval - model.time_shift)
        centered = key_times - key_times.mean()
        slope_per_time = torch.sum(key_positions * centered[None, :, None], dim=1) / torch.sum(centered * centered)
        intercept = key_positions.mean(dim=1) - slope_per_time * key_times.mean()
        displacement = slope_per_time * model.duration
        contribution = k1["s_it"][:, adapter.static_count :].double().cuda().transpose(0, 1)
        numerator = torch.zeros(nd, dtype=torch.float64, device="cuda")
        for index, timestamp in enumerate(samples["T24"]):
            actual = model.get_dynamic_xyz_at_t(timestamp).double()
            linear = intercept + slope_per_time * timestamp
            numerator += contribution[:, index] * torch.sum((actual - linear) ** 2, dim=1)
        score = (numerator / len(samples["T24"])) / (contribution.mean(dim=1) + 1e-12) / (float(k4["length_scale"]) ** 2)
        amplitude = max_motion.double() / float(k4["length_scale"])

        static_floats = 3 + 3 + model._features_dc.shape[1] * model._features_dc.shape[2] + model._features_rest.shape[1] * model._features_rest.shape[2] + 1 + 3 + 4 + 3
        dynamic_floats = model._xyz_motion.shape[1] * 3 + model._features_dc_motion.shape[1] * model._features_dc_motion.shape[2] + model._features_rest_motion.shape[1] * model._features_rest_motion.shape[2] + 3 + 1 + model._opacity_duration_center.shape[1] * model._opacity_duration_center.shape[2] + model._opacity_duration_var.shape[1] * model._opacity_duration_var.shape[2] + model._rotation_motion.shape[1] * 4
        saved_per_point = int((dynamic_floats - static_floats) * 4)
        target_saved = int(math.ceil(0.5 * nd * saved_per_point))
        candidate_count = int(candidate.sum())
        reachable_saved = candidate_count * saved_per_point
        payload = {
            "status": "COMPLETED",
            "dynamic_count": nd,
            "opacity_gate_count": int(opacity_gate.sum()),
            "rotation_gate_count": int(rotation_gate.sum()),
            "candidate_count": candidate_count,
            "candidate_fraction": candidate_count / nd,
            "keyframe_count": int(model._xyz_motion.shape[1]),
            "saved_bytes_per_conversion": saved_per_point,
            "target_saved_bytes": target_saved,
            "reachable_saved_bytes": reachable_saved,
            "target_reachable": reachable_saved >= target_saved,
            "score_candidate_quantiles": torch.quantile(score[candidate], torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], device="cuda", dtype=torch.float64)).cpu().tolist() if candidate_count else [],
            "amplitude_candidate_quantiles": torch.quantile(amplitude[candidate], torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], device="cuda", dtype=torch.float64)).cpu().tolist() if candidate_count else [],
            "seconds": time.perf_counter() - started,
        }
        output = ROOT / "runs" / "research" / "P14" / "probe.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        torch.save({
            "candidate": candidate.cpu(),
            "score": score.cpu(),
            "amplitude": amplitude.cpu(),
            "intercept": intercept.float().cpu(),
            "displacement": displacement.float().cpu(),
            "reference_rotation": reference_q.cpu(),
        }, output.with_name("probe.pt"))
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
