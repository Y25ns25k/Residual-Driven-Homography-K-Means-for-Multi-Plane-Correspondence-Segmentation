"""Tests for normalized DLT homography estimation."""
from __future__ import annotations

import numpy as np
import pytest

from src.geometry import apply_homography
from src.normalized_dlt import HomographyEstimationError, normalize_points, normalized_dlt


def _random_homography(rng: np.random.Generator) -> np.ndarray:
    h = rng.standard_normal((3, 3))
    h[2, 2] = 1.0 + rng.uniform(0.1, 1.0)
    return h / h[2, 2]


def _apply_and_recover(n: int = 20, seed: int = 0, noise: float = 0.0) -> float:
    rng = np.random.default_rng(seed)
    h_gt = _random_homography(rng)
    src = rng.uniform(50, 550, size=(n, 2))
    dst = apply_homography(h_gt, src)
    if noise > 0:
        dst = dst + rng.normal(0, noise, dst.shape)
    h_est = normalized_dlt(src, dst)
    residuals = np.linalg.norm(apply_homography(h_est, src) - dst, axis=1)
    return float(np.mean(residuals))


def test_exact_recovery():
    err = _apply_and_recover(n=30, noise=0.0)
    assert err < 1e-6, f"exact recovery error too large: {err}"


def test_noisy_recovery():
    err = _apply_and_recover(n=100, noise=0.5, seed=7)
    assert err < 1.0, f"noisy recovery error too large: {err}"


def test_weighted_dlt_runs():
    rng = np.random.default_rng(42)
    h_gt = _random_homography(rng)
    src = rng.uniform(50, 550, size=(30, 2))
    dst = apply_homography(h_gt, src)
    weights = rng.uniform(0.5, 1.5, size=30)
    h_est = normalized_dlt(src, dst, weights=weights)
    assert h_est.shape == (3, 3)
    assert abs(h_est[2, 2] - 1.0) < 1e-9


def test_degenerate_raises():
    src = np.ones((4, 2))
    dst = np.ones((4, 2))
    with pytest.raises(HomographyEstimationError):
        normalized_dlt(src, dst)


def test_too_few_points_raises():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (3, 2))
    dst = rng.uniform(0, 100, (3, 2))
    with pytest.raises(HomographyEstimationError):
        normalized_dlt(src, dst)


def test_normalize_points_mean_dist():
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    norm, T = normalize_points(pts)
    mean_dist = float(np.mean(np.linalg.norm(norm, axis=1)))
    assert abs(mean_dist - np.sqrt(2.0)) < 0.1
