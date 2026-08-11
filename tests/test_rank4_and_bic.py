from __future__ import annotations

import numpy as np
import pytest

from src.homography_kmeans.geometry import apply_homography, plane_induced_homography
from src.homography_kmeans.hkm import ResidualHomographyKMeans
from src.homography_kmeans.model_selection import select_models_bic
from src.homography_kmeans.rank4 import denoise_homographies_rank4, homography_stack_singular_values


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


def test_plane_induced_stack_is_rank_four() -> None:
    sv = homography_stack_singular_values(_plane_family(6))
    assert sv[4] / sv[0] < 1e-6


def test_rank4_denoise_noop_for_small_k() -> None:
    hs = _plane_family(3)
    out, info = denoise_homographies_rank4(hs)
    assert info["applied"] is False
    assert all(np.allclose(a, b) for a, b in zip(out, hs))


def test_rank4_denoise_applies_for_k_above_rank() -> None:
    rng = np.random.default_rng(0)
    noisy = [H + rng.normal(0, 1e-4, (3, 3)) for H in _plane_family(6)]
    out, info = denoise_homographies_rank4(noisy)
    assert info["applied"] is True
    assert len(out) == 6
    assert len(info["singular_values"]) == 6
    for H in out:
        assert np.all(np.isfinite(H))


def test_bic_selection_drops_redundant_model() -> None:
    rng = np.random.default_rng(1)
    hs = _plane_family(4)
    H_a, H_b = hs[0], hs[3]
    x1a = rng.uniform(0, 640, (120, 2))
    x1b = rng.uniform(0, 640, (120, 2))
    x1 = np.vstack([x1a, x1b])
    x2 = np.vstack(
        [
            apply_homography(H_a, x1a) + rng.normal(0, 0.5, (120, 2)),
            apply_homography(H_b, x1b) + rng.normal(0, 0.5, (120, 2)),
        ]
    )
    junk = H_a + rng.normal(0, 0.05, (3, 3))
    subset, labels, info = select_models_bic([H_a, H_b, junk], x1, x2, tau_abs=4.0)
    assert info["K_selected"] == 2
    assert len(subset) == 2
    assert labels.shape == (240,)


def test_adaptive_sigma_min_config() -> None:
    cfg = {"hkm": {"tau_abs": 4.0, "sigma_min": 0.75, "adaptive_sigma_min": True}}
    ec = ResidualHomographyKMeans(cfg)._energy_config()
    assert ec.sigma_min == pytest.approx(0.5 * (4.0 / 3.0))
