from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import torch
import cv2

from .config import DEFAULT_MODEL, DEFAULT_SOURCE, ROOT, git_provenance, parse_namespace, sha256_file, sha256_json


def _times(duration: int, count: int) -> list[int]:
    return [int((duration - 1) * index // (count - 1)) for index in range(count)]


def _camera_images(source: Path) -> dict[str, list[Path]]:
    result = {}
    for directory in sorted(source.glob("cam[0-9][0-9]")):
        paths = sorted(directory.glob("*.png"))
        if paths:
            result[directory.name] = paths
    return result


def _camera_media(source: Path) -> dict[str, dict]:
    image_directories = _camera_images(source)
    names = sorted({*image_directories, *(path.stem for path in source.glob("cam[0-9][0-9].mp4"))})
    result = {}
    for name in names:
        video = source / f"{name}.mp4"
        pngs = image_directories.get(name, [])
        if video.exists():
            capture = cv2.VideoCapture(str(video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
        else:
            frame_count = len(pngs)
            width = height = 0
            fps = 0.0
        result[name] = {
            "video": video if video.exists() else None,
            "pngs": pngs,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
        }
    return result


def _inventory(paths: Iterable[Path], hash_images: bool) -> list[dict]:
    rows = []
    for path in paths:
        stat = path.stat()
        row = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if hash_images:
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def build_manifests(output_dir: Path, hash_images: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_namespace(DEFAULT_MODEL / "cfg_args")
    cameras = _camera_media(DEFAULT_SOURCE)
    if "cam00" not in cameras:
        raise RuntimeError("Required final-test camera cam00 is missing")
    development = [name for name in ("cam01", "cam02") if name in cameras]
    if len(development) != 2:
        development = [name for name in sorted(cameras) if name != "cam00"][:2]
    fitting = [name for name in sorted(cameras) if name not in {"cam00", *development}]
    c4_indices = sorted({index * (len(fitting) - 1) // 3 for index in range(4)})
    c4 = [fitting[index] for index in c4_indices]
    media_paths = []
    for media in cameras.values():
        if media["video"] is not None:
            media_paths.append(media["video"])
        else:
            media_paths.extend(media["pngs"])
    image_rows = _inventory(media_paths, hash_images)
    reference = {
        "schema": "research-reference-v1",
        "scene": "cut_roasted_beef",
        "model_path": DEFAULT_MODEL.relative_to(ROOT).as_posix(),
        "source_path": DEFAULT_SOURCE.relative_to(ROOT).as_posix(),
        "iteration": cfg["iterations"],
        "cfg_args_sha256": sha256_file(DEFAULT_MODEL / "cfg_args"),
        "static_ply_sha256": sha256_file(DEFAULT_MODEL / "point_cloud" / "iteration_40000" / "point_cloud.ply"),
        "dynamic_ply_sha256": sha256_file(DEFAULT_MODEL / "point_cloud" / "iteration_40000" / "dynamic_point_cloud.ply"),
        "resolution": cfg["resolution"],
        "duration": cfg["duration"],
        "near": cfg["near"],
        "far": cfg["far"],
        "white_background": cfg["white_background"],
        "model": cfg["model"],
        "interp_type": cfg["interp_type"],
        "rot_interp_type": cfg["rot_interp_type"],
        "image_inventory_hash_mode": "sha256" if hash_images else "metadata",
        "image_inventory_sha256": sha256_json(image_rows),
        "git": git_provenance(),
    }
    splits = {
        "schema": "research-splits-v1",
        "fitting": fitting,
        "development": development,
        "final_test": ["cam00"],
        "c4": c4,
        "camera_frame_counts": {name: media["frame_count"] for name, media in cameras.items()},
        "camera_media": {
            name: {
                "video": media["video"].relative_to(ROOT).as_posix() if media["video"] is not None else None,
                "extracted_png_count": len(media["pngs"]),
                "frame_count": media["frame_count"],
                "width": media["width"],
                "height": media["height"],
                "fps": media["fps"],
            }
            for name, media in cameras.items()
        },
        "media_inventory": image_rows,
        "leakage_note": "The pretrained checkpoint saw non-cam00 views; development is compression validation, not unseen-view validation.",
    }
    samples = {
        "schema": "research-samples-v1",
        "T24": _times(cfg["duration"], 24),
        "T60": _times(cfg["duration"], 60),
        "windows": [[0, 9], [70, 79], [145, 154], [220, 229], [290, 299]],
        "fixed_visualization_times": [0, 74, 149, 224, 299],
        "statistics_resolution": [338, 254],
        "evaluation_resolution": [1352, 1014],
        "patch_size": 8,
        "rays_per_patch_axis": 4,
    }
    environment = {
        "schema": "research-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pid": os.getpid(),
    }
    payloads = {
        "reference.json": reference,
        "splits.json": splits,
        "samples.json": samples,
        "environment.json": environment,
    }
    for name, payload in payloads.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {name: sha256_file(output_dir / name) for name in payloads}
