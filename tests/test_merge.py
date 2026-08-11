from __future__ import annotations

import numpy as np

from src.homography_kmeans.dlt import estimate_homography_dlt
from src.homography_kmeans.energy import EnergyConfig, compute_energy
from src.homography_kmeans.geometry import apply_homography
from src.homography_kmeans.merge import merge_until_stable


def _data():
    rng = np.random.default_rng(2)
    H = np.array([[1.02, 0.01, 8.0], [-0.02, 1.0, 4.0], [1e-4, -1e-4, 1.0]])
    x1 = rng.uniform(0, 220, (50, 2))
    x2 = apply_homography(H, x1) + rng.normal(0, 0.05, (50, 2))
    return x1, x2, H


def test_merge_duplicate_homographies_energy_non_increasing():
    x1, x2, H = _data()
    H1 = estimate_homography_dlt(x1[:25], x2[:25])
    H2 = estimate_homography_dlt(x1[25:], x2[25:])
    labels = np.r_[np.zeros(25, dtype=np.int32), np.ones(25, dtype=np.int32)]
    cfg = EnergyConfig(lambda_K=10.0, gamma_outlier=8.0, tau_abs=3.0)
    old_energy, _, _, _ = compute_energy([H1, H2], x1, x2, labels=labels, scales=np.ones(2), config=cfg)
    Hs_new, labels_new, decisions = merge_until_stable([H1, H2], labels, x1, x2, (240, 320), 2.0, 10, cfg)
    new_energy, _, _, _ = compute_energy(Hs_new, x1, x2, labels=labels_new, scales=np.ones(len(Hs_new)), config=cfg)
    assert len(decisions) == 1
    assert len(Hs_new) == 1
    assert new_energy <= old_energy


def test_merge_does_not_merge_clearly_different_homographies():
    x1, x2, H = _data()
    H_far = np.array([[0.8, 0.1, 60.0], [-0.05, 1.2, -40.0], [5e-4, -4e-4, 1.0]])
    labels = np.zeros(len(x1), dtype=np.int32)
    cfg = EnergyConfig(lambda_K=10.0, gamma_outlier=8.0, tau_abs=3.0)
    Hs_new, labels_new, decisions = merge_until_stable([H, H_far], labels, x1, x2, (240, 320), 2.0, 10, cfg)
    assert len(decisions) == 0
    assert len(Hs_new) == 2
