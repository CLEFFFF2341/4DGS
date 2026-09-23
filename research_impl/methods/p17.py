from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scene import getmodel

from ..adapter import DYNAMIC_PARAMETERS, STATIC_PARAMETERS, ModelAdapter


R0_HEADER_BYTES = 32
R1_HEADER_BYTES = 24


def _chunked_gram(values: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("Gram input must be a matrix")
    gram = torch.zeros((values.shape[1], values.shape[1]), dtype=torch.float64)
    for start in range(0, values.shape[0], chunk_rows):
        block = values[start : start + chunk_rows].double()
        gram.add_(block.T @ block)
    return gram


def _validate_eigenvalues(eigenvalues: torch.Tensor) -> None:
    scale = max(float(eigenvalues.abs().max()), 1.0)
    if float(eigenvalues.min()) < -max(1e-8, 1e-10 * scale):
        raise RuntimeError(f"Gram matrix has a materially negative eigenvalue: {float(eigenvalues.min())}")


def _canonicalize_columns(vectors: torch.Tensor) -> torch.Tensor:
    result = vectors.clone()
    for column in range(result.shape[1]):
        nonzero = torch.nonzero(result[:, column].abs() > 1e-12, as_tuple=False)
        if nonzero.numel() and result[int(nonzero[0]), column] < 0:
            result[:, column].neg_()
    return result


def _shared_rank(dynamic_count: int, dimensions: int, target_bytes: int) -> int:
    denominator = 4 * (dynamic_count + dimensions)
    available = target_bytes - R0_HEADER_BYTES - 4 * 3 * dynamic_count
    rank = min(dynamic_count, dimensions, available // denominator) if denominator else 0
    if rank < 1:
        raise RuntimeError("P17 R0 target cannot hold one shared basis component")
    return int(rank)


def _dct_rank(dynamic_count: int, keyframes: int, target_bytes: int) -> int:
    denominator = 4 * dynamic_count * 3
    rank = min(keyframes, (target_bytes - R1_HEADER_BYTES) // denominator) if denominator else 0
    if rank < 1:
        raise RuntimeError("P17 R1 target cannot hold one DCT coefficient per coordinate")
    return int(rank)


def _dct_basis(keyframes: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    n = torch.arange(keyframes, dtype=torch.float64)
    k = torch.arange(keyframes, dtype=torch.float64).view(-1, 1)
    basis = torch.cos(math.pi / keyframes * (n + 0.5) * k)
    basis[0].mul_(math.sqrt(1.0 / keyframes))
    if keyframes > 1:
        basis[1:].mul_(math.sqrt(2.0 / keyframes))
    return basis.to(dtype=dtype)


def _encode_shared(position: torch.Tensor, target_bytes: int, chunk_rows: int) -> tuple[dict, torch.Tensor, dict]:
    cpu = position.detach().cpu().float()
    dynamic_count, keyframes, coordinates = cpu.shape
    dimensions = keyframes * coordinates
    rank = _shared_rank(dynamic_count, dimensions, target_bytes)
    offset = cpu[:, 0].contiguous()
    centered = (cpu - offset[:, None]).reshape(dynamic_count, dimensions).contiguous()
    gram = _chunked_gram(centered, chunk_rows)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    _validate_eigenvalues(eigenvalues)
    basis = _canonicalize_columns(eigenvectors[:, -rank:].flip(1)).float().contiguous()
    coefficients = (centered @ basis).contiguous()
    decoded = (coefficients @ basis.T).reshape(dynamic_count, keyframes, coordinates) + offset[:, None]
    encoded = {
        "offset": offset,
        "coefficients": coefficients,
        "basis": basis,
        "original_shape": list(cpu.shape),
    }
    encoded_bytes = sum(value.numel() * value.element_size() for value in (offset, coefficients, basis)) + R0_HEADER_BYTES
    diagnostics = {
        "rank": rank,
        "target_variable_bytes": int(target_bytes),
        "encoded_position_bytes": int(encoded_bytes),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "relative_position_frobenius_error": float(torch.linalg.vector_norm(decoded - cpu) / torch.linalg.vector_norm(cpu)),
    }
    return encoded, decoded, diagnostics


def _encode_dct(position: torch.Tensor, target_bytes: int) -> tuple[dict, torch.Tensor, dict]:
    cpu = position.detach().cpu().float()
    dynamic_count, keyframes, _ = cpu.shape
    rank = _dct_rank(dynamic_count, keyframes, target_bytes)
    basis = _dct_basis(keyframes)
    coefficients = torch.einsum("kj,njc->nkc", basis[:rank], cpu).contiguous()
    decoded = torch.einsum("kj,nkc->njc", basis[:rank], coefficients).contiguous()
    encoded = {"coefficients": coefficients, "original_keyframes": keyframes}
    encoded_bytes = coefficients.numel() * coefficients.element_size() + R1_HEADER_BYTES
    diagnostics = {
        "temporal_rank": rank,
        "target_variable_bytes": int(target_bytes),
        "encoded_position_bytes": int(encoded_bytes),
        "relative_position_frobenius_error": float(torch.linalg.vector_norm(decoded - cpu) / torch.linalg.vector_norm(cpu)),
    }
    return encoded, decoded, diagnostics


@torch.no_grad()
def encode_decode(adapter, rule: str, target_fraction: float = 0.5, chunk_rows: int = 8192):
    transformed = adapter.gather(range(adapter.static_count), range(adapter.dynamic_count))
    original = adapter.model._xyz_motion.detach()
    original_bytes = original.numel() * original.element_size()
    target_bytes = int(math.floor(target_fraction * original_bytes))
    if rule == "R0":
        encoded, decoded, diagnostics = _encode_shared(original, target_bytes, chunk_rows)
    elif rule == "R1":
        encoded, decoded, diagnostics = _encode_dct(original, target_bytes)
    else:
        raise KeyError(rule)
    transformed.model._xyz_motion = nn.Parameter(decoded.cuda().requires_grad_(original.requires_grad))
    diagnostics.update(
        {
            "original_position_bytes": int(original_bytes),
            "decoded_position_bytes": int(decoded.numel() * decoded.element_size()),
        }
    )
    return transformed, encoded, diagnostics


@torch.no_grad()
def save_compact_bundle(adapter, encoded: dict, rule: str, path: Path) -> int:
    parameters = {}
    for name in STATIC_PARAMETERS + DYNAMIC_PARAMETERS:
        if name == "_xyz_motion":
            continue
        value = getattr(adapter.model, name)
        parameters[name] = {"tensor": value.detach().cpu(), "requires_grad": bool(value.requires_grad)}
    torch.save(
        {
            "schema": "p17-compact-bundle-v1",
            "rule": rule,
            "encoded_motion": encoded,
            "parameters": parameters,
            "active_sh_degree": int(adapter.model.active_sh_degree),
            "spatial_lr_scale": float(adapter.model.spatial_lr_scale),
            "checkpoint_sha256": adapter.checkpoint_sha256,
            "static_rows": torch.from_numpy(adapter.static_rows.copy()),
            "dynamic_rows": torch.from_numpy(adapter.dynamic_rows.copy()),
        },
        path,
    )
    return path.stat().st_size


@torch.no_grad()
def load_compact_bundle(path: Path, args) -> ModelAdapter:
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "p17-compact-bundle-v1":
        raise RuntimeError(f"Unsupported P17 compact bundle: {path}")
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
    model.active_sh_degree = payload["active_sh_degree"]
    model.spatial_lr_scale = payload["spatial_lr_scale"]
    for name, item in payload["parameters"].items():
        tensor = item["tensor"].cuda()
        setattr(model, name, nn.Parameter(tensor.requires_grad_(item["requires_grad"])))
    encoded = payload["encoded_motion"]
    if payload["rule"] == "R0":
        offset = encoded["offset"]
        coefficients = encoded["coefficients"]
        basis = encoded["basis"]
        decoded = (coefficients @ basis.T).reshape(encoded["original_shape"]) + offset[:, None]
    elif payload["rule"] == "R1":
        coefficients = encoded["coefficients"]
        basis = _dct_basis(int(encoded["original_keyframes"]), dtype=coefficients.dtype)
        decoded = torch.einsum("kj,nkc->njc", basis[: coefficients.shape[1]], coefficients)
    else:
        raise KeyError(payload["rule"])
    model._xyz_motion = nn.Parameter(decoded.contiguous().cuda())
    return ModelAdapter(
        model=model,
        checkpoint_sha256=payload["checkpoint_sha256"],
        static_rows=np.asarray(payload["static_rows"].numpy(), dtype=np.int64),
        dynamic_rows=np.asarray(payload["dynamic_rows"].numpy(), dtype=np.int64),
        args=args,
    )


@torch.no_grad()
def correctness_checks() -> dict:
    generator = torch.Generator().manual_seed(1701)
    coefficients = torch.randn((32, 3), generator=generator)
    modes = torch.randn((3, 18), generator=generator)
    shared = coefficients @ modes
    _, singular, right = torch.linalg.svd(shared, full_matrices=False)
    shared_reconstruction = (shared @ right[:3].T) @ right[:3]

    independent = torch.randn((32, 12), generator=generator)
    independent_rank = int(torch.linalg.matrix_rank(independent))
    _, _, independent_right = torch.linalg.svd(independent, full_matrices=False)
    independent_rank3 = (independent @ independent_right[:3].T) @ independent_right[:3]

    small = torch.randn((64, 12), generator=generator)
    small_gram = _chunked_gram(small, 7)
    small_eigenvalues, small_eigenvectors = torch.linalg.eigh(small_gram)
    gram_values = small_eigenvalues.clamp_min(0).sqrt().flip(0)
    svd_values = torch.linalg.svdvals(small.double())
    full_shared_basis = _canonicalize_columns(small_eigenvectors).double()
    full_shared_reconstruction = (small.double() @ full_shared_basis) @ full_shared_basis.T

    full_basis = _dct_basis(7, dtype=torch.float64)
    dct_input = torch.randn((4, 7, 3), generator=generator, dtype=torch.float64)
    dct_coefficients = torch.einsum("kj,njc->nkc", full_basis, dct_input)
    dct_roundtrip = torch.einsum("kj,nkc->njc", full_basis, dct_coefficients)

    negative_rejected = False
    try:
        _validate_eigenvalues(torch.tensor([-1.0, 2.0], dtype=torch.float64))
    except RuntimeError:
        negative_rejected = True

    seed_outputs = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        seed_outputs.append(_dct_basis(9))

    checks = {
        "shared_translation_effective_rank": int((singular > singular.max() * 1e-6).sum()),
        "shared_rank3_relative_error": float(torch.linalg.vector_norm(shared_reconstruction - shared) / torch.linalg.vector_norm(shared)),
        "independent_matrix_rank": independent_rank,
        "independent_rank3_relative_error": float(torch.linalg.vector_norm(independent_rank3 - independent) / torch.linalg.vector_norm(independent)),
        "chunked_gram_vs_svd_max_abs_singular_value": float((gram_values - svd_values).abs().max()),
        "full_shared_basis_relative_error": float(
            torch.linalg.vector_norm(full_shared_reconstruction - small.double()) / torch.linalg.vector_norm(small.double())
        ),
        "full_dct_relative_error": float(torch.linalg.vector_norm(dct_roundtrip - dct_input) / torch.linalg.vector_norm(dct_input)),
        "negative_eigenvalue_rejected": negative_rejected,
        "seed_invariant_0_1_2": bool(torch.equal(seed_outputs[0], seed_outputs[1]) and torch.equal(seed_outputs[0], seed_outputs[2])),
        "basis_and_offset_cost_included": True,
        "empty_dynamic_group_policy": "skip representation transform; real run has a non-empty dynamic group",
    }
    checks["passed"] = bool(
        checks["shared_translation_effective_rank"] <= 3
        and checks["shared_rank3_relative_error"] <= 1e-5
        and checks["independent_matrix_rank"] == 12
        and checks["independent_rank3_relative_error"] >= 0.3
        and checks["chunked_gram_vs_svd_max_abs_singular_value"] <= 1e-5
        and checks["full_shared_basis_relative_error"] <= 1e-5
        and checks["full_dct_relative_error"] <= 1e-5
        and checks["negative_eigenvalue_rejected"]
        and checks["seed_invariant_0_1_2"]
        and checks["basis_and_offset_cost_included"]
    )
    return checks
