from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import diff_gaussian_rasterization_df._C as raster_extension

from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim

from .adapter import load_bundle
from .config import sha256_file, sha256_json
from .evaluate import VideoFrameDecoder, load_ground_truth, pipeline, render_one, select_camera


def _set_position_lrs(model, checkpoint_iteration: int) -> dict[str, float]:
    values = {}
    for group in model.optimizer.param_groups:
        if group["name"] == "xyz":
            group["lr"] = model.xyz_scheduler_args(checkpoint_iteration)
        elif group["name"] == "motion_xyz":
            group["lr"] = model.xyz_motion_scheduler_args(checkpoint_iteration)
        values[group["name"]] = float(group["lr"])
    return values


def finetune_smoke(adapter, scene, camera_names: list[str], times: list[int], output: Path) -> dict:
    run_specification = {
        "schema": "finetune-smoke-v1",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "camera_names": camera_names,
        "times": times,
        "steps": 10,
        "extension_sha256": sha256_file(Path(raster_extension.__file__)),
    }
    run_key = sha256_json(run_specification)
    output = output / run_key
    result_path = output / "finetune_smoke.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["reused"] = True
        return result
    output.mkdir(parents=True, exist_ok=False)
    model = adapter.gather(range(adapter.static_count), range(adapter.dynamic_count))
    model.model.spatial_lr_scale = float(scene.cameras_extent)
    model.model.training_setup(adapter.args)
    learning_rates = _set_position_lrs(model.model, int(adapter.args.iterations))
    original_xyz = model.model._xyz.detach().clone()
    original_dynamic_xyz = model.model._xyz_motion.detach().clone()
    initial_counts = [model.static_count, model.dynamic_count]
    decoders: dict[Path, VideoFrameDecoder] = {}
    losses = []
    step_times = []
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for step in range(10):
        name = camera_names[step % len(camera_names)]
        timestamp = times[step]
        camera = select_camera(scene, name, timestamp)
        gt = load_ground_truth(camera, decoders)
        started = time.perf_counter()
        rendered = render(
            camera,
            model.model,
            pipeline(),
            background,
            timestamp=timestamp,
            training=True,
            near=adapter.args.near,
            far=adapter.args.far,
        )["render"]
        loss = 0.8 * l1_loss(rendered, gt) + 0.2 * (1.0 - ssim(rendered, gt))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite finetune loss at step {step}")
        loss.backward()
        if model.model._opacity_duration_var.grad is not None:
            model.model._opacity_duration_var.grad.nan_to_num_()
        model.model.optimizer.step()
        model.model.optimizer.zero_grad(set_to_none=True)
        step_times.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
        del rendered, gt, loss
    for decoder in decoders.values():
        decoder.close()
    if [model.static_count, model.dynamic_count] != initial_counts:
        raise RuntimeError("Point counts changed during fixed-budget finetune smoke")
    bundle_dir = output / "bundle"
    if not bundle_dir.exists():
        bundle_metadata = model.save_bundle(bundle_dir)
    else:
        bundle_metadata = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    restored = load_bundle(bundle_dir, adapter.args)
    check_camera = select_camera(scene, camera_names[0], times[-1])
    reload_max_abs = float((render_one(model, check_camera) - render_one(restored, check_camera)).abs().max())
    result = {
        "status": "COMPLETED",
        "run_key": run_key,
        "specification": run_specification,
        "steps": 10,
        "losses": losses,
        "step_seconds": step_times,
        "learning_rates": learning_rates,
        "counts_before": initial_counts,
        "counts_after": [model.static_count, model.dynamic_count],
        "static_xyz_max_abs_update": float((model.model._xyz.detach() - original_xyz).abs().max()),
        "dynamic_xyz_max_abs_update": float((model.model._xyz_motion.detach() - original_dynamic_xyz).abs().max()),
        "reload_max_abs": reload_max_abs,
        "reload_passed": reload_max_abs <= 1e-6,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "bundle": bundle_metadata,
        "reused": False,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
