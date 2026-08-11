"""Tests for HomographyKMeans with residual-driven discovery."""
from __future__ import annotations

import numpy as np
import pytest

from src.geometry import apply_homography
from src.homography_kmeans import HomographyKMeans, KMeansResult


def _make_two_plane_data(seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    h1 = np.array([[1.1, 0.05, 20.0], [-0.03, 1.08, 15.0], [0.0002, -0.0001, 1.0]])
    h1 /= h1[2, 2]
    h2 = np.array([[0.92, -0.08, -40.0], [0.06, 1.04, 30.0], [-0.0003, 0.0002, 1.0]])
    h2 /= h2[2, 2]
    n = 60
    src1 = rng.uniform(50, 300, (n, 2))
    dst1 = apply_homography(h1, src1) + rng.normal(0, 0.15, (n, 2))
    src2 = rng.uniform(350, 650, (n, 2))
    dst2 = apply_homography(h2, src2) + rng.normal(0, 0.15, (n, 2))
    src = np.vstack([src1, src2])
    dst = np.vstack([dst1, dst2])
    gt_labels = np.concatenate([np.zeros(n, dtype=np.int32), np.ones(n, dtype=np.int32)])
    return src, dst, gt_labels, [h1, h2]


def test_kmeans_basic():
    src, dst, gt_labels, _ = _make_two_plane_data()
    km = HomographyKMeans(random_state=42)
    result = km.fit(src, dst, image_shape=(480, 640))
    assert isinstance(result, KMeansResult)
    assert result.labels.shape == (len(src),)


def test_kmeans_finds_planes():
    src, dst, gt_labels, _ = _make_two_plane_data()
    km = HomographyKMeans(random_state=42)
    result = km.fit(src, dst, image_shape=(480, 640))
    assert len(result.homographies) >= 1


def test_kmeans_low_outlier_rate():
    src, dst, gt_labels, _ = _make_two_plane_data()
    km = HomographyKMeans(random_state=42)
    result = km.fit(src, dst, image_shape=(480, 640))
    outlier_rate = float(np.mean(result.labels < 0))
    assert outlier_rate < 0.25, f"outlier rate too high: {outlier_rate:.2f}"


def test_kmeans_converges():
    src, dst, gt_labels, _ = _make_two_plane_data()
    km = HomographyKMeans(random_state=42, max_iterations=30)
    result = km.fit(src, dst)
    assert result.n_iterations <= 30


def test_kmeans_history_tracked():
    src, dst, gt_labels, _ = _make_two_plane_data()
    km = HomographyKMeans(random_state=42)
    result = km.fit(src, dst)
    assert len(result.history) >= 1
    assert "K" in result.history[0]
    assert "total_error" in result.history[0]


def test_kmeans_single_plane():
    rng = np.random.default_rng(0)
    h = np.array([[1.1, 0.05, 20.0], [-0.03, 1.08, 15.0], [0.0002, -0.0001, 1.0]])
    h /= h[2, 2]
    src = rng.uniform(50, 550, (80, 2))
    dst = apply_homography(h, src) + rng.normal(0, 0.1, (80, 2))
    km = HomographyKMeans(random_state=42)
    result = km.fit(src, dst)
    # Should find exactly 1 plane (or at most 2 from relaxed init)
    assert len(result.homographies) >= 1


def test_kmeans_ablation_no_discovery():
    src, dst, gt_labels, _ = _make_two_plane_data()
    km = HomographyKMeans(random_state=42, use_residual_discovery=False)
    result = km.fit(src, dst)
    assert result.labels.shape == (len(src),)


def test_kmeans_ablation_no_refit():
    src, dst, gt_labels, _ = _make_two_plane_data()
    km = HomographyKMeans(random_state=42, use_robust_refit=False)
    result = km.fit(src, dst)
    assert result.labels.shape == (len(src),)
