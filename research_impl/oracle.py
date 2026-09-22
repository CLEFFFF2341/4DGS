from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def composite(colors: np.ndarray, alphas: np.ndarray, background: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64)
    alphas = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 0.99)
    out = np.zeros(3, dtype=np.float64)
    transmittance = 1.0
    for color, alpha in zip(colors, alphas):
        if alpha < 1.0 / 255.0:
            continue
        out += transmittance * alpha * color
        transmittance *= 1.0 - alpha
        if transmittance < 1e-4:
            break
    return out + transmittance * np.asarray(background, dtype=np.float64)


def analytic_deletions(colors: np.ndarray, alphas: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Return C_full-C_without_i for a ray without renderer early termination."""
    colors = np.asarray(colors, dtype=np.float64)
    alphas = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 0.99)
    n = len(alphas)
    suffix = np.empty((n + 1, 3), dtype=np.float64)
    suffix[n] = np.asarray(background, dtype=np.float64)
    for index in range(n - 1, -1, -1):
        alpha = 0.0 if alphas[index] < 1.0 / 255.0 else alphas[index]
        suffix[index] = alpha * colors[index] + (1.0 - alpha) * suffix[index + 1]
    transmittance = 1.0
    deletion = np.zeros((n, 3), dtype=np.float64)
    for index in range(n):
        alpha = 0.0 if alphas[index] < 1.0 / 255.0 else alphas[index]
        deletion[index] = transmittance * alpha * (colors[index] - suffix[index + 1])
        transmittance *= 1.0 - alpha
    return deletion


def brute_force_deletions(colors: np.ndarray, alphas: np.ndarray, background: np.ndarray) -> np.ndarray:
    full = composite_no_early_stop(colors, alphas, background)
    values = []
    for index in range(len(alphas)):
        keep = np.ones(len(alphas), dtype=bool)
        keep[index] = False
        values.append(full - composite_no_early_stop(colors[keep], alphas[keep], background))
    return np.asarray(values, dtype=np.float64).reshape((-1, 3))


def composite_no_early_stop(colors: np.ndarray, alphas: np.ndarray, background: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64)
    alphas = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 0.99)
    out = np.zeros(3, dtype=np.float64)
    transmittance = 1.0
    for color, alpha in zip(colors, alphas):
        if alpha < 1.0 / 255.0:
            continue
        out += transmittance * alpha * color
        transmittance *= 1.0 - alpha
    return out + transmittance * np.asarray(background, dtype=np.float64)


def run_oracle_suite(output: Path | None = None) -> dict:
    cases = {
        "zero_points": (np.empty((0, 3)), np.empty((0,)), np.array([0.2, 0.3, 0.4])),
        "one_point": (np.array([[0.8, 0.1, 0.2]]), np.array([0.4]), np.zeros(3)),
        "same_color": (np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]]), np.array([0.8, 0.7]), np.zeros(3)),
        "occluded_reveal": (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.99, 0.9]), np.zeros(3)),
        "transparent": (np.array([[1.0, 1.0, 1.0], [0.2, 0.3, 0.4]]), np.array([1e-4, 0.5]), np.zeros(3)),
        "alpha_clip": (np.array([[1.0, 0.2, 0.1]]), np.array([1.5]), np.array([0.1, 0.2, 0.3])),
        "non_black_bg": (np.array([[0.7, 0.5, 0.1]]), np.array([0.25]), np.array([0.1, 0.4, 0.9])),
        "signed_cancel": (np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]), np.array([0.5, 0.5]), np.array([0.5, 0.5, 0.5])),
    }
    details = {}
    passed = True
    for name, (colors, alphas, background) in cases.items():
        analytic = analytic_deletions(colors, alphas, background)
        brute = brute_force_deletions(colors, alphas, background)
        error = float(np.max(np.abs(analytic - brute))) if analytic.size else 0.0
        case_passed = error <= 1e-12
        details[name] = {"max_abs_error": error, "passed": case_passed}
        passed &= case_passed
    result = {"schema": "alpha-oracle-v1", "passed": passed, "cases": details}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
