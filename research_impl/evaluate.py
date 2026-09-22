from __future__ import annotations

import csv
import copy
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image

from gaussian_renderer import render
from lpipsPyTorch.modules.lpips import LPIPS
from utils.general_utils import PILtoTorch
from utils.image_utils import psnr
from utils.loss_utils import ssim


def camera_name(camera: Any) -> str:
    return Path(camera.image_path).parent.name


def select_camera(scene: Any, name: str, timestamp: int) -> Any:
    pools = list(scene.train_cameras[1.0]) + list(scene.test_cameras[1.0])
    matches = [camera for camera in pools if camera_name(camera) == name and int(camera.timestamp) == int(timestamp)]
    if len(matches) == 1:
        return matches[0]
    base = [camera for camera in pools if camera_name(camera) == name]
    if len(base) != 1:
        raise RuntimeError(f"Expected one base camera for {name}, got {len(base)}")
    # The local N3V preprocessing contains one pose-bearing PNG for train views,
    # while the complete temporal signal remains in the read-only MP4.  Camera
    # pose is fixed for N3V, so clone the audited pose and attach the requested
    # video frame without mutating Scene or the source dataset.
    camera = copy.copy(base[0])
    video = Path(camera.image_path).parents[1] / f"{name}.mp4"
    if not video.exists():
        raise RuntimeError(f"No frame {timestamp} PNG and no source video: {video}")
    camera.timestamp = int(timestamp)
    camera.image_name = f"{timestamp:06d}.png"
    camera.image_path = str(video)
    camera._research_frame_index = int(timestamp)
    return camera


class VideoFrameDecoder:
    def __init__(self, path: Path):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        self.next_index = 0

    def read(self, index: int) -> Image.Image:
        if index < self.next_index:
            self.capture.release()
            self.capture = cv2.VideoCapture(str(self.path))
            self.next_index = 0
        frame = None
        while self.next_index <= index:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"Unable to decode frame {index} from {self.path}")
            self.next_index += 1
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        self.capture.release()


def load_ground_truth(camera: Any, decoders: dict[Path, VideoFrameDecoder] | None = None) -> torch.Tensor:
    path = Path(camera.image_path)
    if path.suffix.lower() == ".mp4":
        if decoders is None:
            decoder = VideoFrameDecoder(path)
            image_source = decoder.read(int(camera._research_frame_index))
            decoder.close()
        else:
            if path not in decoders:
                decoders[path] = VideoFrameDecoder(path)
            decoder = decoders[path]
            image_source = decoder.read(int(camera._research_frame_index))
    else:
        image_source = Image.open(path)
    image = PILtoTorch(image_source, camera.resolution)[:3, ...]
    return (image / camera.im_scale).clamp(0, 1).cuda()


def pipeline() -> SimpleNamespace:
    return SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)


def render_one(adapter: Any, camera: Any) -> torch.Tensor:
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    return render(
        camera,
        adapter.model,
        pipeline(),
        background,
        timestamp=camera.timestamp,
        near=adapter.args.near,
        far=adapter.args.far,
    )["render"].clamp(0, 1)


@torch.no_grad()
def smoke_render(adapter: Any, scene: Any, camera_names: list[str], times: list[int], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for name in camera_names:
        for timestamp in times:
            camera = select_camera(scene, name, timestamp)
            start = time.perf_counter()
            image = render_one(adapter, camera)
            elapsed = time.perf_counter() - start
            rows.append({
                "camera": name,
                "timestamp": timestamp,
                "width": int(image.shape[2]),
                "height": int(image.shape[1]),
                "minimum": float(image.min()),
                "maximum": float(image.max()),
                "mean": float(image.mean()),
                "seconds": elapsed,
            })
    result = {
        "status": "COMPLETED",
        "renders": rows,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (output / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def evaluate(adapter: Any, scene: Any, camera_names: list[str], times: list[int], output: Path) -> dict:
    if "cam00" in camera_names:
        raise RuntimeError("Final test evaluation is sealed until research/progress.json freezes the final-test manifest")
    output.mkdir(parents=True, exist_ok=True)
    lpips_model = LPIPS("alex", "0.1").cuda().eval()
    rows = []
    decoders: dict[Path, VideoFrameDecoder] = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for name in camera_names:
        for timestamp in times:
            camera = select_camera(scene, name, timestamp)
            gt = load_ground_truth(camera, decoders)
            render_start = time.perf_counter()
            image = render_one(adapter, camera)
            render_seconds = time.perf_counter() - render_start
            mse = torch.mean((image - gt) ** 2)
            row = {
                "camera": name,
                "timestamp": timestamp,
                "psnr": float(psnr(image[None], gt[None]).item()),
                "ssim": float(ssim(image[None], gt[None]).item()),
                "lpips_alex": float(lpips_model(image[None], gt[None]).item()),
                "gt_mse": float(mse.item()),
                "render_seconds": render_seconds,
            }
            rows.append(row)
            del gt, image
    for decoder in decoders.values():
        decoder.close()
    with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "COMPLETED",
        "count": len(rows),
        "mean_psnr": statistics.fmean(row["psnr"] for row in rows),
        "mean_ssim": statistics.fmean(row["ssim"] for row in rows),
        "mean_lpips_alex": statistics.fmean(row["lpips_alex"] for row in rows),
        "mean_gt_mse": statistics.fmean(row["gt_mse"] for row in rows),
        "mean_render_seconds": statistics.fmean(row["render_seconds"] for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def evaluate_against_reference(
    adapter: Any,
    reference: Any,
    scene: Any,
    camera_names: list[str],
    times: list[int],
    output: Path,
) -> dict:
    """Evaluate a method against GT and its immutable teacher in one pass."""
    if "cam00" in camera_names:
        raise RuntimeError("Final test evaluation is sealed until research/progress.json freezes the final-test manifest")
    output.mkdir(parents=True, exist_ok=True)
    lpips_model = LPIPS("alex", "0.1").cuda().eval()
    rows = []
    decoders: dict[Path, VideoFrameDecoder] = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for name in camera_names:
        for timestamp in times:
            camera = select_camera(scene, name, timestamp)
            gt = load_ground_truth(camera, decoders)
            render_start = time.perf_counter()
            image = render_one(adapter, camera)
            render_seconds = time.perf_counter() - render_start
            teacher = render_one(reference, camera)
            method_mse = torch.mean((image - gt) ** 2)
            teacher_mse = torch.mean((teacher - gt) ** 2)
            method_psnr = float(psnr(image[None], gt[None]).item())
            teacher_psnr = float(psnr(teacher[None], gt[None]).item())
            rows.append({
                "camera": name,
                "timestamp": timestamp,
                "psnr": method_psnr,
                "ssim": float(ssim(image[None], gt[None]).item()),
                "lpips_alex": float(lpips_model(image[None], gt[None]).item()),
                "gt_mse": float(method_mse.item()),
                "teacher_mse": float(torch.mean((image - teacher) ** 2).item()),
                "delta_mse": float((method_mse - teacher_mse).item()),
                "reference_psnr": teacher_psnr,
                "psnr_drop": teacher_psnr - method_psnr,
                "render_seconds": render_seconds,
            })
            del gt, image, teacher
    for decoder in decoders.values():
        decoder.close()
    with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tail_count = max(1, int(np.ceil(0.1 * len(rows))))
    drop_descending = sorted(rows, key=lambda row: (-row["psnr_drop"], row["camera"], row["timestamp"]))
    worst = drop_descending[0]
    last_rows = [row for row in rows if row["timestamp"] >= 270]
    summary = {
        "status": "COMPLETED",
        "count": len(rows),
        "mean_psnr": statistics.fmean(row["psnr"] for row in rows),
        "mean_ssim": statistics.fmean(row["ssim"] for row in rows),
        "mean_lpips_alex": statistics.fmean(row["lpips_alex"] for row in rows),
        "mean_gt_mse": statistics.fmean(row["gt_mse"] for row in rows),
        "mean_teacher_mse": statistics.fmean(row["teacher_mse"] for row in rows),
        "mean_delta_mse": statistics.fmean(row["delta_mse"] for row in rows),
        "mean_psnr_drop": statistics.fmean(row["psnr_drop"] for row in rows),
        "worst_10pct_mean_psnr_drop": statistics.fmean(row["psnr_drop"] for row in drop_descending[:tail_count]),
        "p95_delta_mse": float(np.percentile([row["delta_mse"] for row in rows], 95)),
        "worst_frame": {"camera": worst["camera"], "timestamp": worst["timestamp"], "psnr_drop": worst["psnr_drop"]},
        "last_30_mean_psnr": statistics.fmean(row["psnr"] for row in last_rows),
        "last_30_mean_delta_mse": statistics.fmean(row["delta_mse"] for row in last_rows),
        "mean_render_seconds": statistics.fmean(row["render_seconds"] for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def save_comparison_visualizations(
    adapter: Any,
    reference: Any,
    scene: Any,
    camera_names: list[str],
    times: list[int],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    decoders: dict[Path, VideoFrameDecoder] = {}
    for name in camera_names:
        for timestamp in times:
            camera = select_camera(scene, name, timestamp)
            gt = load_ground_truth(camera, decoders)
            teacher = render_one(reference, camera)
            image = render_one(adapter, camera)
            panel = torch.cat((gt, teacher, image), dim=2).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(panel).save(output / f"{name}_{timestamp:03d}_gt-reference-method.png")
    for decoder in decoders.values():
        decoder.close()


@torch.no_grad()
def benchmark_latency(adapter: Any, scene: Any, inputs: list[tuple[str, int]], output: Path) -> dict:
    cameras = [select_camera(scene, name, timestamp) for name, timestamp in inputs]
    for index in range(20):
        render_one(adapter, cameras[index % len(cameras)])
    blocks = []
    torch.cuda.reset_peak_memory_stats()
    for block_index in range(3):
        wall_values = []
        gpu_values = []
        for index in range(100):
            camera = cameras[index % len(cameras)]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start_event.record()
            render_one(adapter, camera)
            end_event.record()
            torch.cuda.synchronize()
            wall_values.append(time.perf_counter() - wall_start)
            gpu_values.append(start_event.elapsed_time(end_event) / 1000.0)
        blocks.append({
            "block": block_index,
            "wall_mean_seconds": statistics.fmean(wall_values),
            "wall_p50_seconds": float(np.percentile(wall_values, 50)),
            "wall_p95_seconds": float(np.percentile(wall_values, 95)),
            "gpu_mean_seconds": statistics.fmean(gpu_values),
            "gpu_p50_seconds": float(np.percentile(gpu_values, 50)),
            "gpu_p95_seconds": float(np.percentile(gpu_values, 95)),
        })
    result = {
        "status": "COMPLETED",
        "warmup": 20,
        "iterations_per_block": 100,
        "inputs": [{"camera": name, "timestamp": timestamp} for name, timestamp in inputs],
        "blocks": blocks,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def retention_check(adapter: Any, scene: Any, camera_name_value: str, timestamp: int, output: Path) -> dict:
    camera = select_camera(scene, camera_name_value, timestamp)
    reference = render_one(adapter, camera)
    subset = adapter.gather(range(adapter.static_count), range(adapter.dynamic_count))
    retained = render_one(subset, camera)
    difference = (reference - retained).abs()
    result = {
        "status": "COMPLETED",
        "camera": camera_name_value,
        "timestamp": timestamp,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "passed": bool(float(difference.max()) <= 1e-6),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
