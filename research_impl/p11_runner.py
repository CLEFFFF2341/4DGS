from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import imageio_ffmpeg
from PIL import Image

from gaussian_renderer import render
from utils.general_utils import PILtoTorch
from utils.loss_utils import l1_loss, ssim

from .adapter import DYNAMIC_PARAMETERS, STATIC_PARAMETERS, ModelAdapter, load_bundle, load_reference
from .config import ROOT, git_provenance, sha256_file
from .evaluate import (
    benchmark_latency,
    evaluate_against_reference,
    pipeline,
    render_one,
    save_comparison_visualizations,
    select_camera,
)
from .finetune import _set_position_lrs
from .methods.p11 import (
    hard_group_mask,
    initial_logits,
    selection_from_probabilities,
    two_point_recovery_toy,
)
from .runner import gpu_lock


P02_ROOT = ROOT / "runs" / "research" / "P02" / "cut_roasted_beef" / "b0.5" / "R0"
OUTPUT_ROOT = ROOT / "runs" / "research" / "P11" / "cut_roasted_beef" / "b0.5"
TRAINING_CACHE = ROOT / "research_cache" / "p11_training_gt_v1"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def find_p01_r2_bundle() -> Path:
    root = ROOT / "runs" / "research" / "P01" / "cut_roasted_beef" / "b0.5" / "seed0"
    matches = []
    for path in root.glob("*/resolved_config.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("rule") == "R2":
            matches.append(path.parent / "bundle")
    if len(matches) != 1 or not matches[0].exists():
        raise RuntimeError(f"Expected exactly one P01.R2 bundle, got {matches}")
    return matches[0]


def make_schedule(seed: int, camera_names: list[str], steps: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    camera_indices = rng.integers(0, len(camera_names), size=steps)
    timestamps = rng.integers(0, 300, size=steps)
    return [
        {"step": index, "camera": camera_names[int(camera_indices[index])], "timestamp": int(timestamps[index])}
        for index in range(steps)
    ]


def training_frame_path(name: str, timestamp: int) -> Path:
    return TRAINING_CACHE / name / f"{timestamp:06d}.png"


def _verified_training_frame(path: Path) -> bool:
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        return False
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    return bool(
        metadata.get("schema") == "p11-training-frame-v3"
        and metadata.get("decoder_backend") == "imageio-ffmpeg"
        and metadata.get("png_sha256") == sha256_file(path)
    )


def _decode_ffmpeg_frames(scene: Any, name: str, timestamps: list[int]) -> dict[int, np.ndarray]:
    base_camera = select_camera(scene, name, 0)
    source = ROOT / "data" / "cut_roasted_beef" / f"{name}.mp4"
    if not source.exists():
        raise FileNotFoundError(source)
    requested = set(timestamps)
    reader = imageio_ffmpeg.read_frames(
        str(source),
        pix_fmt="rgb24",
        input_params=["-threads", "1"],
    )
    metadata = next(reader)
    width, height = metadata["size"]
    result: dict[int, np.ndarray] = {}
    try:
        for index, payload in enumerate(reader):
            if index in requested:
                source_image = Image.frombytes("RGB", (width, height), payload)
                resized = PILtoTorch(source_image, base_camera.resolution)[:3, ...]
                resized = (resized / base_camera.im_scale).clamp(0, 1)
                result[index] = resized.mul(255).round().byte().permute(1, 2, 0).numpy()
            if index >= max(timestamps):
                break
    finally:
        reader.close()
    if set(result) != requested:
        raise RuntimeError(f"FFmpeg did not return every requested frame for {name}: {sorted(requested - set(result))}")
    return result


def prepare_training_frames(scene: Any, schedule: list[dict[str, Any]]) -> dict[str, Any]:
    requested: dict[str, set[int]] = {}
    for item in schedule:
        requested.setdefault(item["camera"], set()).add(int(item["timestamp"]))
    created = []
    started = time.perf_counter()
    for name in sorted(requested):
        missing = [
            timestamp
            for timestamp in sorted(requested[name])
            if not _verified_training_frame(training_frame_path(name, timestamp))
        ]
        if not missing:
            continue
        accepted = None
        for attempt in range(1, 6):
            passes = [_decode_ffmpeg_frames(scene, name, missing) for _ in range(2)]
            if all(np.array_equal(passes[0][timestamp], passes[1][timestamp]) for timestamp in missing):
                accepted = (passes[0], attempt)
                break
        if accepted is None:
            raise RuntimeError(f"Repeated sequential video decodes disagree for {name}: {missing}")
        decoded, attempt = accepted
        for timestamp in missing:
            image = decoded[timestamp]
            path = training_frame_path(name, timestamp)
            previous = None
            if path.with_suffix(".json").exists():
                previous = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            pixel_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(path)
            dump(
                path.with_suffix(".json"),
                {
                    "schema": "p11-training-frame-v3",
                    "camera": name,
                    "timestamp": timestamp,
                    "decoder_backend": "imageio-ffmpeg",
                    "decoder_threads": 1,
                    "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(),
                    "verification": "two independent sequential FFmpeg decodes are byte-identical",
                    "attempt": attempt,
                    "pixel_sha256": pixel_sha256,
                    "png_sha256": sha256_file(path),
                    "superseded_schema": previous.get("schema") if previous else None,
                    "superseded_pixel_match": previous.get("pixel_sha256") == pixel_sha256 if previous else None,
                },
            )
            created.append(str(path.relative_to(ROOT)))
    return {
        "requested_unique_frames": int(sum(len(values) for values in requested.values())),
        "created_frames": len(created),
        "created_paths": created,
        "seconds": time.perf_counter() - started,
    }


def load_cached_training_gt(camera: Any, name: str, timestamp: int) -> torch.Tensor:
    path = training_frame_path(name, timestamp)
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(path)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if sha256_file(path) != metadata["png_sha256"]:
        raise RuntimeError(f"Training-frame cache checksum mismatch: {path}")
    with Image.open(path) as source:
        image = PILtoTorch(source, camera.resolution)[:3, ...]
    return image.cuda()


def freeze_model(adapter: ModelAdapter) -> None:
    for name in STATIC_PARAMETERS + DYNAMIC_PARAMETERS:
        parameter = getattr(adapter.model, name)
        parameter.requires_grad_(False)
        parameter.grad = None


def train_gate(
    reference: ModelAdapter,
    scene: Any,
    scores: np.ndarray,
    schedule: list[dict[str, Any]],
    rule: str,
    config: dict[str, Any],
) -> tuple[ModelAdapter, torch.Tensor, dict[str, Any]]:
    if rule not in {"R0", "R1"}:
        raise KeyError(rule)
    model = reference.gather(range(reference.static_count), range(reference.dynamic_count))
    freeze_model(model)
    logits = torch.nn.Parameter(initial_logits(scores))
    optimizer = torch.optim.Adam([logits], lr=float(config["gate_learning_rate"]))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    history = []
    hard_counts = []
    step_times = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step, item in enumerate(schedule):
        camera = select_camera(scene, item["camera"], item["timestamp"])
        gt = load_cached_training_gt(camera, item["camera"], item["timestamp"])
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.sigmoid(logits)
        if rule == "R0":
            hard = hard_group_mask(probabilities, model, float(config["budget_fraction"]))
            gate = (hard - probabilities.detach() + probabilities).float()
            penalty = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
            hard_counts.append(int(hard.sum().item()))
        else:
            gate = probabilities.float()
            total = int(np.floor(float(config["budget_fraction"]) * model.total_count))
            keep_static = int(np.floor(total * model.static_count / model.total_count))
            keep_dynamic = total - keep_static
            static_penalty = ((probabilities[: model.static_count].sum() - keep_static) / model.static_count) ** 2
            dynamic_penalty = ((probabilities[model.static_count :].sum() - keep_dynamic) / model.dynamic_count) ** 2
            penalty = float(config["soft_budget_weight"]) * (static_penalty + dynamic_penalty)
        step_started = time.perf_counter()
        rendered = render(
            camera,
            model.model,
            pipeline(),
            background,
            timestamp=item["timestamp"],
            training=True,
            near=model.args.near,
            far=model.args.far,
            mask_gate=gate,
        )["render"]
        image_loss = 0.8 * l1_loss(rendered, gt) + 0.2 * (1.0 - ssim(rendered, gt))
        loss = image_loss + penalty
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {rule} loss at step {step}")
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise RuntimeError(f"Invalid gate gradient at step {step}")
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_started)
        history.append(
            {
                "step": step,
                "camera": item["camera"],
                "timestamp": item["timestamp"],
                "image_loss": float(image_loss.detach()),
                "budget_penalty": float(penalty.detach()),
                "total_loss": float(loss.detach()),
                "probability_mean": float(probabilities.detach().mean()),
                "logit_grad_l1": float(logits.grad.detach().abs().mean()),
            }
        )
        del gt, rendered, image_loss, loss, probabilities, gate, penalty
    probabilities = torch.sigmoid(logits.detach())
    frozen_parameters = [getattr(model.model, name) for name in STATIC_PARAMETERS + DYNAMIC_PARAMETERS]
    frozen_ok = all(not parameter.requires_grad and parameter.grad is None for parameter in frozen_parameters)
    result = {
        "rule": rule,
        "steps": len(schedule),
        "history": history,
        "step_seconds": step_times,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "hard_count_min": min(hard_counts) if hard_counts else None,
        "hard_count_max": max(hard_counts) if hard_counts else None,
        "all_non_gate_parameters_frozen_without_grad": frozen_ok,
    }
    if not frozen_ok:
        raise RuntimeError("A frozen model parameter received a gradient")
    return model, probabilities, result


def train_r2(
    source: ModelAdapter,
    scene: Any,
    schedule: list[dict[str, Any]],
) -> tuple[ModelAdapter, dict[str, Any]]:
    model = source.gather(range(source.static_count), range(source.dynamic_count))
    model.model.spatial_lr_scale = float(scene.cameras_extent)
    model.model.training_setup(model.args)
    learning_rates = _set_position_lrs(model.model, int(model.args.iterations))
    initial_counts = [model.static_count, model.dynamic_count]
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    history = []
    step_times = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step, item in enumerate(schedule):
        camera = select_camera(scene, item["camera"], item["timestamp"])
        gt = load_cached_training_gt(camera, item["camera"], item["timestamp"])
        step_started = time.perf_counter()
        rendered = render(
            camera,
            model.model,
            pipeline(),
            background,
            timestamp=item["timestamp"],
            training=True,
            near=model.args.near,
            far=model.args.far,
        )["render"]
        loss = 0.8 * l1_loss(rendered, gt) + 0.2 * (1.0 - ssim(rendered, gt))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite R2 loss at step {step}")
        loss.backward()
        if model.model._opacity_duration_var.grad is not None:
            model.model._opacity_duration_var.grad.nan_to_num_()
        model.model.optimizer.step()
        model.model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_started)
        history.append(
            {
                "step": step,
                "camera": item["camera"],
                "timestamp": item["timestamp"],
                "total_loss": float(loss.detach()),
            }
        )
        del gt, rendered, loss
    if [model.static_count, model.dynamic_count] != initial_counts:
        raise RuntimeError("R2 changed point counts during fixed-budget finetuning")
    return model, {
        "rule": "R2",
        "steps": len(schedule),
        "history": history,
        "step_seconds": step_times,
        "learning_rates": learning_rates,
        "counts_before": initial_counts,
        "counts_after": [model.static_count, model.dynamic_count],
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def renderer_correctness(reference: ModelAdapter, scene: Any, scores: np.ndarray) -> dict[str, Any]:
    camera = select_camera(scene, "cam01", 0)
    original = render_one(reference, camera)
    ones = torch.ones(reference.total_count, dtype=torch.float32, device="cuda")
    all_one = render_one(reference, camera, mask_gate=ones)
    one_difference = (original - all_one).abs()

    zero = torch.zeros(reference.total_count, dtype=torch.float32, device="cuda", requires_grad=True)
    zero_image = render_one(reference, camera, mask_gate=zero)
    zero_objective = zero_image.mean()
    zero_objective.backward()
    point = int(zero.grad.abs().argmax())
    analytic = float(zero.grad[point])
    epsilon = 1e-3
    perturbation = torch.zeros_like(zero.detach())
    perturbation[point] = epsilon
    numeric_image = render_one(reference, camera, mask_gate=perturbation)
    numeric = float((numeric_image.mean() - zero_objective.detach()) / epsilon)

    probabilities = torch.sigmoid(initial_logits(scores))
    hard_a = hard_group_mask(probabilities, reference, 0.5)
    hard_b = hard_group_mask(probabilities, reference, 0.5)
    p02_ids = np.load(P02_ROOT / "kept_ids.npz")
    selected = selection_from_probabilities(probabilities, reference, 0.5)
    checks = {
        "all_one_max_abs": float(one_difference.max()),
        "all_one_mean_abs": float(one_difference.mean()),
        "all_one_equivalent": bool(float(one_difference.max()) <= 1e-6),
        "zero_gate_render_mean": float(zero_objective),
        "zero_gate_nonzero_gradient_count": int((zero.grad != 0).sum()),
        "finite_difference_point": point,
        "finite_difference_analytic": analytic,
        "finite_difference_numeric": numeric,
        "finite_difference_same_direction": bool(analytic * numeric > 0),
        "finite_difference_relative_error": abs(analytic - numeric) / (abs(numeric) + 1e-12),
        "masked_points_survive_opacity_threshold": bool(int((zero.grad != 0).sum()) > 0),
        "hard_count": int(hard_a.sum()),
        "hard_count_exact": int(hard_a.sum()) == 124447,
        "hard_repeat_identical": bool(torch.equal(hard_a, hard_b)),
        "initial_hard_matches_p02": bool(
            np.array_equal(selected.static_indices, p02_ids["static_indices"])
            and np.array_equal(selected.dynamic_indices, p02_ids["dynamic_indices"])
        ),
        "two_point_recovery_toy": two_point_recovery_toy(),
    }
    checks["passed"] = bool(
        checks["all_one_equivalent"]
        and checks["finite_difference_same_direction"]
        and checks["finite_difference_relative_error"] <= 0.02
        and checks["masked_points_survive_opacity_threshold"]
        and checks["hard_count_exact"]
        and checks["hard_repeat_identical"]
        and checks["initial_hard_matches_p02"]
        and checks["two_point_recovery_toy"]["passed"]
    )
    return checks


def run_smoke(
    reference: ModelAdapter,
    scene: Any,
    scores: np.ndarray,
    p01_r2: ModelAdapter,
    fitting: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    output = OUTPUT_ROOT / "smoke"
    schedule = make_schedule(0, fitting, 10)
    cache = prepare_training_frames(scene, schedule)
    rows = {}
    for rule in config["rules"]:
        started = time.perf_counter()
        if rule in {"R0", "R1"}:
            model, probabilities, training = train_gate(reference, scene, scores, schedule, rule, config)
            selection = selection_from_probabilities(probabilities, reference, float(config["budget_fraction"]))
            rows[rule] = {
                "training": training,
                "selection": selection.diagnostics,
                "final_probabilities_finite": bool(torch.isfinite(probabilities).all()),
            }
            del model, probabilities
        else:
            model, training = train_r2(p01_r2, scene, schedule)
            rows[rule] = {"training": training, "selection": {"actual_total": model.total_count}}
            del model
        rows[rule]["wall_seconds"] = time.perf_counter() - started
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "status": "COMPLETED",
        "training_decoder": {"backend": "imageio-ffmpeg", "version": imageio_ffmpeg.get_ffmpeg_version()},
        "steps_per_rule": 10,
        "schedule": schedule,
        "training_cache": cache,
        "rules": rows,
    }
    dump(output / "smoke.json", result)
    return result


def run_formal_row(
    reference: ModelAdapter,
    scene: Any,
    scores: np.ndarray,
    p01_r2: ModelAdapter,
    splits: dict[str, Any],
    samples: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    rule: str,
) -> dict[str, Any]:
    output = OUTPUT_ROOT / f"seed{seed}" / rule
    status_path = output / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETED":
            return json.loads((output / "result.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    dump(status_path, {"status": "RUNNING", "rule": rule, "seed": seed})
    schedule = make_schedule(seed, splits["fitting"], int(config["steps"]))
    dump(output / "sampling_schedule.json", schedule)
    cache = prepare_training_frames(scene, schedule)
    started = time.perf_counter()
    soft_summary = None
    gate_state = None
    if rule in {"R0", "R1"}:
        full_model, probabilities, training = train_gate(reference, scene, scores, schedule, rule, config)
        selection = selection_from_probabilities(probabilities, reference, float(config["budget_fraction"]))
        if rule == "R1":
            soft_summary = evaluate_against_reference(
                full_model,
                reference,
                scene,
                splits["development"],
                samples["T24"],
                output / "soft_evaluation",
                mask_gate=probabilities.float(),
            )
        final_model = reference.gather(selection.static_indices, selection.dynamic_indices)
        gate_state = {
            "probabilities": probabilities.cpu(),
            "static_indices": torch.from_numpy(selection.static_indices),
            "dynamic_indices": torch.from_numpy(selection.dynamic_indices),
        }
        torch.save(gate_state, output / "gate_state.pt")
        dump(output / "selection.json", selection.diagnostics)
        np.savez_compressed(
            output / "kept_ids.npz",
            static_indices=selection.static_indices,
            dynamic_indices=selection.dynamic_indices,
            static_rows=reference.static_rows[selection.static_indices],
            dynamic_rows=reference.dynamic_rows[selection.dynamic_indices],
        )
        del full_model, probabilities
    else:
        final_model, training = train_r2(p01_r2, scene, schedule)
        selection = None
    bundle_dir = output / "bundle"
    if not bundle_dir.exists():
        bundle_metadata = final_model.save_bundle(bundle_dir)
    else:
        bundle_metadata = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    summary = evaluate_against_reference(
        final_model,
        reference,
        scene,
        splits["development"],
        samples["T24"],
        output / "evaluation",
    )
    if soft_summary is not None:
        soft_summary["soft_to_hard_psnr_drop"] = soft_summary["mean_psnr"] - summary["mean_psnr"]
        soft_summary["soft_to_hard_lpips_increase"] = summary["mean_lpips_alex"] - soft_summary["mean_lpips_alex"]
        dump(output / "soft_to_hard.json", soft_summary)
    inputs = [(splits["development"][index % 2], samples["T24"][index]) for index in range(10)]
    latency = benchmark_latency(final_model, scene, inputs, output / "latency.json")
    if seed == 0:
        save_comparison_visualizations(
            final_model,
            reference,
            scene,
            splits["development"],
            samples["fixed_visualization_times"],
            output / "visualizations",
        )
    total_seconds = time.perf_counter() - started
    resolved = {**config, "seed": seed, "rule": rule, "p01_r2_bundle": str(find_p01_r2_bundle())}
    dump(output / "resolved_config.json", resolved)
    dump(output / "training.json", training)
    dump(
        output / "resources.json",
        {
            "actual_total_points": final_model.total_count,
            "static_points": final_model.static_count,
            "dynamic_points": final_model.dynamic_count,
            "bundle_bytes": directory_bytes(bundle_dir),
            "training_cache": cache,
            "latency": latency,
            "total_seconds": total_seconds,
        },
    )
    dump(
        output / "provenance.json",
        {
            "git": git_provenance(),
            "checkpoint_sha256": reference.checkpoint_sha256,
            "p02_selection_sha256": sha256_file(P02_ROOT / "selection.pt"),
            "raster_extension_sha256": sha256_file(Path(__import__("diff_gaussian_rasterization_df")._C.__file__)),
            "config_sha256": sha256_file(ROOT / "configs" / "research" / "p11.json"),
        },
    )
    (output / "commands.txt").write_text(
        f"python -m research_impl.p11_runner --seed {seed} --rule {rule}\n", encoding="utf-8"
    )
    result = {
        "status": "COMPLETED",
        "rule": rule,
        "seed": seed,
        "summary": summary,
        "soft_summary": soft_summary,
        "training_final_loss": training["history"][-1]["total_loss"],
        "bundle": bundle_metadata,
        "total_seconds": total_seconds,
    }
    dump(output / "result.json", result)
    dump(status_path, {"status": "COMPLETED", "rule": rule, "seed": seed})
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_rule.setdefault(result["rule"], []).append(result)
    p02 = json.loads((P02_ROOT / "summary.json").read_text(encoding="utf-8"))
    output: dict[str, Any] = {"status": "COMPLETED", "p02_zero_ft": p02, "rules": {}}
    for rule, rows in by_rule.items():
        rows = sorted(rows, key=lambda value: value["seed"])
        psnr_values = [row["summary"]["mean_psnr"] for row in rows]
        ssim_values = [row["summary"]["mean_ssim"] for row in rows]
        lpips_values = [row["summary"]["mean_lpips_alex"] for row in rows]
        tail_values = [row["summary"]["worst_10pct_mean_psnr_drop"] for row in rows]
        output["rules"][rule] = {
            "seeds": [row["seed"] for row in rows],
            "runs": rows,
            "mean_psnr": statistics.fmean(psnr_values),
            "sample_std_psnr": statistics.stdev(psnr_values) if len(psnr_values) > 1 else 0.0,
            "mean_ssim": statistics.fmean(ssim_values),
            "sample_std_ssim": statistics.stdev(ssim_values) if len(ssim_values) > 1 else 0.0,
            "mean_lpips_alex": statistics.fmean(lpips_values),
            "sample_std_lpips_alex": statistics.stdev(lpips_values) if len(lpips_values) > 1 else 0.0,
            "mean_worst_10pct_psnr_drop": statistics.fmean(tail_values),
            "delta_psnr_vs_p02_zero_ft": statistics.fmean(psnr_values) - p02["mean_psnr"],
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--rule", choices=["R0", "R1", "R2"])
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    config_path = ROOT / "configs" / "research" / "p11.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "research" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "research" / "manifests" / "samples.json").read_text(encoding="utf-8"))
    p02_state = torch.load(P02_ROOT / "selection.pt", map_location="cpu")
    scores = np.asarray(p02_state["scores"], dtype=np.float64)
    with gpu_lock():
        reference, scene = load_reference(load_cameras=True)
        p01_r2 = load_bundle(find_p01_r2_bundle(), reference.args)
        correctness_path = OUTPUT_ROOT / "correctness_checks.json"
        if correctness_path.exists():
            correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
            if not correctness.get("passed", False):
                correctness = renderer_correctness(reference, scene, scores)
                dump(correctness_path, correctness)
        else:
            correctness = renderer_correctness(reference, scene, scores)
            dump(correctness_path, correctness)
        if not correctness["passed"]:
            raise RuntimeError(correctness)
        smoke_path = OUTPUT_ROOT / "smoke" / "smoke.json"
        if smoke_path.exists():
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            if smoke.get("training_decoder", {}).get("backend") != "imageio-ffmpeg":
                smoke = run_smoke(reference, scene, scores, p01_r2, splits["fitting"], config)
        else:
            smoke = run_smoke(reference, scene, scores, p01_r2, splits["fitting"], config)
        if cli.smoke_only:
            print(json.dumps({"correctness": correctness, "smoke": smoke}, indent=2))
            return
        seeds = [cli.seed] if cli.seed is not None else config["seeds"]
        rules = [cli.rule] if cli.rule is not None else config["rules"]
        results = []
        for seed in seeds:
            for rule in rules:
                results.append(run_formal_row(reference, scene, scores, p01_r2, splits, samples, config, seed, rule))
                gc.collect()
                torch.cuda.empty_cache()
    if cli.seed is None and cli.rule is None:
        combined = aggregate(results)
        dump(OUTPUT_ROOT / "aggregate.json", combined)
        print(json.dumps(combined, indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
