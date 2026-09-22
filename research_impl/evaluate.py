from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
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
    if len(matches) != 1:
        raise RuntimeError(f"Expected one camera for {name}@{timestamp}, got {len(matches)}")
    return matches[0]


def load_ground_truth(camera: Any) -> torch.Tensor:
    image = PILtoTorch(Image.open(camera.image_path), camera.resolution)[:3, ...]
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
        timestamp=int(camera.timestamp),
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
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for name in camera_names:
        for timestamp in times:
            camera = select_camera(scene, name, timestamp)
            gt = load_ground_truth(camera)
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
