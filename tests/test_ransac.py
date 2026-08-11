"""Tests for RANSAC and sequential RANSAC."""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from src.geometry import apply_homography
from src.normalized_dlt import normalized_dlt
from src.ransac import RansacResult, SequentialRansacResult, estimate_homography_ransac, sequential_ransac
from src.homography_kmeans.ransac import estimate_homography_ransac as estimate_new_ransac
from src.homography_kmeans.sequential import sequential_ransac as sequential_new_ransac
from src.homography_kmeans.synthetic import generate_synthetic_scene
from src.homography_kmeans.experiment import _method_fit


def _make_inlier_set(n: int = 80, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    h_gt = np.array([[1.1, 0.05, 20.0], [-0.03, 1.08, 15.0], [0.0002, -0.0001, 1.0]])
    h_gt /= h_gt[2, 2]
    src = rng.uniform(50, 550, size=(n, 2))
    dst = apply_homography(h_gt, src) + rng.normal(0, 0.2, (n, 2))
    return src, dst, h_gt


def test_ransac_finds_homography():
    src, dst, h_gt = _make_inlier_set(80)
    result = estimate_homography_ransac(src, dst, threshold=2.0, min_inliers=4)
    assert result.success
    assert result.homography is not None
    assert result.n_inliers >= 20


def test_ransac_with_outliers():
    rng = np.random.default_rng(0)
    src_in, dst_in, h_gt = _make_inlier_set(60)
    out_src = rng.uniform(0, 640, (30, 2))
    out_dst = rng.uniform(0, 480, (30, 2))
    src = np.vstack([src_in, out_src])
    dst = np.vstack([dst_in, out_dst])
    result = estimate_homography_ransac(src, dst, threshold=2.0, min_inliers=10)
    assert result.success
    assert result.n_inliers >= 40


def test_sequential_ransac_two_planes():
    rng = np.random.default_rng(7)
    h1 = np.array([[1.1, 0.05, 20.0], [-0.03, 1.08, 15.0], [0.0002, -0.0001, 1.0]])
    h1 /= h1[2, 2]
    h2 = np.array([[0.95, -0.1, -30.0], [0.08, 1.05, 25.0], [-0.0003, 0.0002, 1.0]])
    h2 /= h2[2, 2]
    src1 = rng.uniform(50, 250, (60, 2))
    dst1 = apply_homography(h1, src1) + rng.normal(0, 0.3, (60, 2))
    src2 = rng.uniform(300, 600, (60, 2))
    dst2 = apply_homography(h2, src2) + rng.normal(0, 0.3, (60, 2))
    src = np.vstack([src1, src2])
    dst = np.vstack([dst1, dst2])
    result = sequential_ransac(src, dst, threshold=1.0, min_inliers=20)
    assert len(result.homographies) >= 1


def test_ransac_too_few_points():
    src = np.ones((3, 2))
    dst = np.ones((3, 2))
    result = estimate_homography_ransac(src, dst)
    assert not result.success


def test_sequential_ransac_labels():
    src, dst, h_gt = _make_inlier_set(80)
    result = sequential_ransac(src, dst, threshold=1.0, min_inliers=20)
    assert isinstance(result, SequentialRansacResult)
    if result.homographies:
        assert result.labels.shape == (80,)
        assert np.any(result.labels >= 0)


def test_new_ransac_recovers_homography_with_outliers():
    rng = np.random.default_rng(13)
    src_in, dst_in, _ = _make_inlier_set(70, seed=13)
    src = np.vstack([src_in, rng.uniform(0, 640, (30, 2))])
    dst = np.vstack([dst_in, rng.uniform(0, 480, (30, 2))])
    result = estimate_new_ransac(src, dst, threshold=2.5, min_support=25, random_state=1)
    assert result.success
    assert result.n_inliers >= 55


def test_new_sequential_ransac_easy_synthetic_returns_model():
    scene = generate_synthetic_scene("easy", "easy", points_per_plane=35, noise_std=0.2, outlier_ratio=0.05, seed=9)
    result = sequential_new_ransac(scene.x1, scene.x2, threshold=3.0, min_support=20, random_state=4)
    assert len(result.homographies) >= 1
    assert result.labels.shape == scene.gt_labels.shape


def test_global_ransac_method_fit_returns_single_model():
    rng = np.random.default_rng(15)
    src_in, dst_in, _ = _make_inlier_set(60, seed=15)
    scene = SimpleNamespace(
        scene_id="single_h",
        x1=np.vstack([src_in, rng.uniform(0, 640, (15, 2))]),
        x2=np.vstack([dst_in, rng.uniform(0, 480, (15, 2))]),
        image_shape=(480, 640),
    )
    config = {
        "image_shape": [480, 640],
        "ransac": {"threshold": 3.0, "max_iterations": 1000, "confidence": 0.999, "min_support": 20},
        "hkm": {"min_support": 20},
    }
    homographies, labels, residuals, runtime, diagnostics = _method_fit(scene, "global_ransac", config, seed=3)
    assert len(homographies) == 1
    assert set(np.unique(labels)).issubset({-1, 0})
    assert np.sum(labels == 0) >= 40
    assert residuals.shape == labels.shape
    assert diagnostics["success"] == 1.0
