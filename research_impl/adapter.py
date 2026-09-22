from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from scene import Scene, getmodel

from .config import DEFAULT_MODEL, DEFAULT_SOURCE, reference_args, sha256_file


STATIC_PARAMETERS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
    "_xyz_disp",
)
DYNAMIC_PARAMETERS = (
    "_xyz_motion",
    "_features_dc_motion",
    "_features_rest_motion",
    "_scaling_motion",
    "_opacity_motion",
    "_opacity_duration_center",
    "_opacity_duration_var",
    "_rotation_motion",
)
STATIC_BUFFERS = (
    "max_radii2D",
    "min_radii2D",
    "xyz_gradient_accum",
    "denom",
    "xyz_error_accum",
    "xyz_error_min",
    "xyz_error_min_timestamp",
    "xyz_ssim_error_accum",
    "error_denom",
)
DYNAMIC_BUFFERS = (
    "motion_max_radii2D",
    "motion_min_radii2D",
    "motion_xyz_gradient_accum",
    "motion_denom",
    "motion_xyz_error_min",
    "motion_xyz_error_mean",
    "motion_xyz_error_min_timestamp",
    "motion_xyz_ssim_error_accum",
    "motion_error_denom",
)


@dataclass
class ModelAdapter:
    model: Any
    checkpoint_sha256: str
    static_rows: np.ndarray
    dynamic_rows: np.ndarray
    args: Any

    @property
    def static_count(self) -> int:
        return int(self.model._xyz.shape[0])

    @property
    def dynamic_count(self) -> int:
        return int(self.model._xyz_motion.shape[0])

    @property
    def total_count(self) -> int:
        return self.static_count + self.dynamic_count

    def stable_id(self, kind: str, row: int) -> str:
        source_row = self.static_rows[row] if kind == "static" else self.dynamic_rows[row]
        return f"{self.checkpoint_sha256}:{kind}:{int(source_row)}"

    def tensor_audit(self) -> dict[str, Any]:
        result: dict[str, Any] = {"static_count": self.static_count, "dynamic_count": self.dynamic_count, "tensors": {}}
        known = set(STATIC_PARAMETERS + DYNAMIC_PARAMETERS + STATIC_BUFFERS + DYNAMIC_BUFFERS)
        for name, value in vars(self.model).items():
            if isinstance(value, torch.Tensor):
                if name in STATIC_PARAMETERS + STATIC_BUFFERS:
                    kind = "static"
                elif name in DYNAMIC_PARAMETERS + DYNAMIC_BUFFERS:
                    kind = "dynamic"
                else:
                    kind = "unclassified"
                result["tensors"][name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "parameter": isinstance(value, nn.Parameter),
                    "kind": kind,
                    "known": name in known,
                }
        unknown_per_point = []
        for name, meta in result["tensors"].items():
            shape = meta["shape"]
            if meta["kind"] == "unclassified" and shape and shape[0] in {self.static_count, self.dynamic_count}:
                unknown_per_point.append(name)
        result["unknown_per_point_tensors"] = unknown_per_point
        return result

    def gather(self, static_indices: Iterable[int], dynamic_indices: Iterable[int]) -> "ModelAdapter":
        sidx = torch.as_tensor(list(static_indices), dtype=torch.long, device="cuda")
        didx = torch.as_tensor(list(dynamic_indices), dtype=torch.long, device="cuda")
        cls = getmodel(self.args.model)
        subset = cls(
            self.args.sh_degree,
            self.args.duration,
            self.args.time_interval,
            self.args.time_pad,
            interp_type=self.args.interp_type,
            rot_interp_type=self.args.rot_interp_type,
            time_pad_type=self.args.time_pad_type,
            var_pad=self.args.var_pad,
            kernel_size=self.args.kernel_size,
        )
        subset.active_sh_degree = self.model.active_sh_degree
        subset.spatial_lr_scale = self.model.spatial_lr_scale
        for name in STATIC_PARAMETERS:
            source = getattr(self.model, name)
            setattr(subset, name, nn.Parameter(source.index_select(0, sidx).detach().clone().requires_grad_(source.requires_grad)))
        for name in DYNAMIC_PARAMETERS:
            source = getattr(self.model, name)
            setattr(subset, name, nn.Parameter(source.index_select(0, didx).detach().clone().requires_grad_(source.requires_grad)))
        for name in STATIC_BUFFERS:
            source = getattr(self.model, name)
            if source.ndim and source.shape[0] == self.static_count:
                setattr(subset, name, source.index_select(0, sidx).detach().clone())
        for name in DYNAMIC_BUFFERS:
            source = getattr(self.model, name)
            if source.ndim and source.shape[0] == self.dynamic_count:
                setattr(subset, name, source.index_select(0, didx).detach().clone())
        return ModelAdapter(
            model=subset,
            checkpoint_sha256=self.checkpoint_sha256,
            static_rows=self.static_rows[sidx.cpu().numpy()],
            dynamic_rows=self.dynamic_rows[didx.cpu().numpy()],
            args=self.args,
        )

    def save_bundle(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=False)
        ply_path = directory / "point_cloud.ply"
        self.model.save_ply(str(ply_path))
        np.savez(
            directory / "stable_ids.npz",
            static_rows=self.static_rows,
            dynamic_rows=self.dynamic_rows,
        )
        metadata = {
            "schema": "research-bundle-v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "static_count": self.static_count,
            "dynamic_count": self.dynamic_count,
            "point_cloud_sha256": sha256_file(ply_path),
            "dynamic_point_cloud_sha256": sha256_file(directory / "dynamic_point_cloud.ply"),
        }
        (directory / "bundle.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata


def load_reference(
    model_path: Path = DEFAULT_MODEL,
    source_path: Path = DEFAULT_SOURCE,
    load_cameras: bool = True,
) -> tuple[ModelAdapter, Scene | None]:
    args = reference_args(model_path, source_path)
    cls = getmodel(args.model)
    model = cls(
        args.sh_degree,
        args.duration,
        args.time_interval,
        args.time_pad,
        interp_type=args.interp_type,
        rot_interp_type=args.rot_interp_type,
        time_pad_type=args.time_pad_type,
        var_pad=args.var_pad,
        kernel_size=args.kernel_size,
    )
    if load_cameras:
        scene = Scene(args, model, load_iteration=args.iterations, shuffle=False, load_cameras=True)
    else:
        model.load_ply(str(model_path / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"))
        scene = None
    static_hash = sha256_file(model_path / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply")
    dynamic_hash = sha256_file(model_path / "point_cloud" / f"iteration_{args.iterations}" / "dynamic_point_cloud.ply")
    checkpoint_hash = f"{static_hash}+{dynamic_hash}"
    adapter = ModelAdapter(
        model=model,
        checkpoint_sha256=checkpoint_hash,
        static_rows=np.arange(model._xyz.shape[0], dtype=np.int64),
        dynamic_rows=np.arange(model._xyz_motion.shape[0], dtype=np.int64),
        args=args,
    )
    return adapter, scene
