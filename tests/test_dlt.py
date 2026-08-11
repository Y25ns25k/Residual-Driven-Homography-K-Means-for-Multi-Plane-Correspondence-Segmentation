from __future__ import annotations

import numpy as np

from src.homography_kmeans.dlt import estimate_homography_dlt
from src.homography_kmeans.geometry import apply_homography, normalize_homography, symmetric_transfer_error


def test_dlt_recovers_known_homography_noiseless():
    rng = np.random.default_rng(3)
    H = normalize_homography(np.array([[1.1, -0.04, 25.0], [0.05, 0.97, -12.0], [2e-4, -1e-4, 1.0]]))
    x1 = rng.uniform(20, 400, (60, 2))
    x2 = apply_homography(H, x1)
    H_est = estimate_homography_dlt(x1, x2)
    assert float(np.median(symmetric_transfer_error(H_est, x1, x2))) < 1e-7


def test_weighted_dlt_runs_and_normalizes():
    rng = np.random.default_rng(4)
    H = normalize_homography(np.array([[0.96, 0.02, -8.0], [-0.04, 1.05, 10.0], [-1e-4, 2e-4, 1.0]]))
    x1 = rng.uniform(0, 300, (30, 2))
    x2 = apply_homography(H, x1)
    weights = rng.uniform(0.3, 1.2, len(x1))
    H_est = estimate_homography_dlt(x1, x2, weights=weights)
    assert H_est.shape == (3, 3)
    assert abs(H_est[2, 2] - 1.0) < 1e-9
