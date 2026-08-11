from __future__ import annotations

import numpy as np

from src.homography_kmeans.geometry import (
    apply_homography,
    functional_warp_distance,
    normalize_points_2d,
    symmetric_transfer_error,
)


def test_symmetric_transfer_error_perfect_correspondences():
    rng = np.random.default_rng(0)
    H = np.array([[1.02, 0.03, 12.0], [-0.02, 0.98, -8.0], [1e-4, -2e-4, 1.0]])
    x1 = rng.uniform(0, 300, (40, 2))
    x2 = apply_homography(H, x1)
    err = symmetric_transfer_error(H, x1, x2)
    assert float(np.max(err)) < 1e-8


def test_normalize_points_2d_mean_distance():
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    norm, T = normalize_points_2d(pts)
    assert T.shape == (3, 3)
    assert abs(float(np.mean(np.linalg.norm(norm, axis=1))) - np.sqrt(2.0)) < 1e-9


def test_functional_distance_scaled_equivalent_homographies():
    H = np.array([[1.0, 0.02, 5.0], [0.01, 0.98, -2.0], [1e-4, 2e-4, 1.0]])
    assert functional_warp_distance(H, 3.0 * H, image_shape=(120, 160)) < 1e-9
