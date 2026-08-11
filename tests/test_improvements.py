from __future__ import annotations

import numpy as np

from src.homography_kmeans.energy import EnergyConfig, error_matrix
from src.homography_kmeans.geometry import apply_homography, plane_induced_homography
from src.homography_kmeans.ransac import estimate_homography_ransac
from src.homography_kmeans.rank4 import rank4_candidate_consistent, rank4_tail_score
from src.homography_kmeans.spatial import icm_smooth_labels, knn_edges


def _plane_family(n_planes: int) -> list[np.ndarray]:
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    theta = 0.05
    R = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    t = np.array([0.3, 0.05, 0.02])
    out = []
    for i in range(n_planes):
        n = np.array([0.3 * np.sin(i), 0.2 * np.cos(i), 1.0])
        out.append(plane_induced_homography(K, R, t, n / np.linalg.norm(n), 3.0 + 0.5 * i))
    return out


def test_local_sampling_finds_spatially_compact_minority_structure() -> None:
    rng = np.random.default_rng(3)
    H = _plane_family(1)[0]
    # 30 plane points clustered in one image corner + 270 gross outliers.
    x1_plane = rng.uniform(0, 120, (30, 2))
    x2_plane = apply_homography(H, x1_plane) + rng.normal(0, 0.5, (30, 2))
    x1_junk = rng.uniform(0, 640, (270, 2))
    x2_junk = rng.uniform(0, 640, (270, 2))
    x1 = np.vstack([x1_plane, x1_junk])
    x2 = np.vstack([x2_plane, x2_junk])
    found = 0
    for seed in range(5):
        res = estimate_homography_ransac(
            x1, x2, threshold=2.5, max_iterations=800, min_support=20,
            random_state=seed, local_sampling=True, local_k=40,
        )
        if res.success and res.n_inliers >= 20:
            found += 1
    assert found >= 3  # local sampling should usually recover the compact plane


def test_icm_smoothing_fixes_isolated_label_flips() -> None:
    rng = np.random.default_rng(4)
    hs = _plane_family(2)
    x1a = rng.uniform(0, 300, (80, 2))
    x1b = rng.uniform(340, 640, (80, 2))
    x1 = np.vstack([x1a, x1b])
    x2 = np.vstack(
        [
            apply_homography(hs[0], x1a) + rng.normal(0, 0.3, (80, 2)),
            apply_homography(hs[1], x1b) + rng.normal(0, 0.3, (80, 2)),
        ]
    )
    errors = error_matrix(hs, x1, x2)
    gt = np.array([0] * 80 + [1] * 80, dtype=np.int32)
    noisy = gt.copy()
    flip = rng.choice(160, size=12, replace=False)
    noisy[flip] = 1 - noisy[flip]
    cfg = EnergyConfig(tau_abs=1e9, tau_norm=1e9)  # isolate the Potts effect
    out = icm_smooth_labels(errors, np.ones(2), noisy, knn_edges(x1, k=8), cfg, lambda_s=0.5)
    assert int(np.sum(out != gt)) <= int(np.sum(noisy != gt))


def test_rank4_tail_score_trivial_below_min_planes() -> None:
    score, trivial = rank4_tail_score(_plane_family(4))
    assert trivial and score == 0.0


def test_rank4_prior_accepts_consistent_candidate() -> None:
    hs = _plane_family(7)
    ok, metrics = rank4_candidate_consistent(hs[:6], hs[6])
    assert ok
    assert metrics["rank4_after"] < 0.35
