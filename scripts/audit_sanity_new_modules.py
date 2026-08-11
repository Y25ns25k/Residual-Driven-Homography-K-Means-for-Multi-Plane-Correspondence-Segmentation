"""Sanity checks for rank4 denoising, BIC selection, and adaptive sigma_min."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.energy import EnergyConfig
from homography_kmeans.geometry import apply_homography, plane_induced_homography
from homography_kmeans.hkm import ResidualHomographyKMeans
from homography_kmeans.model_selection import select_models_bic
from homography_kmeans.rank4 import denoise_homographies_rank4, homography_stack_singular_values

rng = np.random.default_rng(0)

# Build 6 plane-induced homographies sharing K, R, t (true rank-4 family).
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
theta = 0.05
R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
t = np.array([0.3, 0.05, 0.02])
H_true = []
for i in range(6):
    n = np.array([0.3 * np.sin(i), 0.2 * np.cos(i), 1.0])
    n /= np.linalg.norm(n)
    H_true.append(plane_induced_homography(K, R, t, n, 3.0 + 0.5 * i))
sv = homography_stack_singular_values(H_true)
print("singular values of clean 6-plane stack:", np.round(sv, 6))
assert sv[4] / sv[0] < 1e-6, "clean plane-induced stack should be rank <= 4"

H_noisy = [H + rng.normal(0, 1e-3, (3, 3)) for H in H_true]
den, info = denoise_homographies_rank4(H_noisy)
assert info["applied"] is True and len(den) == 6
pts = rng.uniform(0, 640, (200, 2))
err_noisy = np.mean([np.linalg.norm(apply_homography(Hn, pts) - apply_homography(Ht, pts)) for Hn, Ht in zip(H_noisy, H_true)])
err_den = np.mean([np.linalg.norm(apply_homography(Hd, pts) - apply_homography(Ht, pts)) for Hd, Ht in zip(den, H_true)])
print(f"warp error vs GT: noisy={err_noisy:.4f}, denoised={err_den:.4f}")

# K <= 4 must be a no-op.
den3, info3 = denoise_homographies_rank4(H_noisy[:3])
assert info3["applied"] is False
assert all(np.allclose(a, b) for a, b in zip(den3, H_noisy[:3]))
print("K<=4 no-op: ok")

# BIC selection: 2 true planes + 1 junk duplicate model should select K=2.
H_a, H_b = H_true[0], H_true[3]
x1a = rng.uniform(0, 640, (120, 2))
x1b = rng.uniform(0, 640, (120, 2))
x2a = apply_homography(H_a, x1a) + rng.normal(0, 0.5, (120, 2))
x2b = apply_homography(H_b, x1b) + rng.normal(0, 0.5, (120, 2))
x1 = np.vstack([x1a, x1b])
x2 = np.vstack([x2a, x2b])
H_junk = H_a + rng.normal(0, 0.05, (3, 3))
subset, labels, info = select_models_bic([H_a, H_b, H_junk], x1, x2, tau_abs=4.0, tau_norm=3.0, sigma_min=0.75)
print("BIC values:", [(d["K"], round(d["BIC"], 1)) for d in info["bic_values"]])
print(f"BIC selected K={info['K_selected']} (expect 2)")
assert info["K_selected"] == 2

# Adaptive sigma_min flows through the config.
cfg = {"hkm": {"tau_abs": 4.0, "tau_norm": 3.0, "sigma_min": 0.75, "adaptive_sigma_min": True, "min_support": 20}}
km = ResidualHomographyKMeans(cfg)
ec: EnergyConfig = km._energy_config()
print(f"adaptive sigma_min in EnergyConfig: {ec.sigma_min:.4f} (expect {0.5 * (4.0 / 3.0):.4f})")
assert abs(ec.sigma_min - 0.5 * (4.0 / 3.0)) < 1e-12

print("\nall sanity checks passed")
