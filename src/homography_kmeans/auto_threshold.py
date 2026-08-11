"""Scene-adaptive threshold calibration from a robust pilot fit.

The production config assumes a nominal noise level of sigma = 1 px
(RANSAC threshold 2.5 px, tau_abs 4.0 px, sigma_min 0.75 px). Real scenes
deviate from that nominal level, and a single global threshold is then too
tight for noisy scenes and too loose for clean ones. This module estimates a
per-scene noise scale sigma_hat from the inlier residuals of one loose pilot
RANSAC fit and scales the pixel thresholds linearly:

    sigma_hat = 1.4826 * MAD(pilot inlier residuals), clipped to [lo, hi]
    threshold_scene = threshold_default * sigma_hat

Only the largest-support single homography is used for the pilot, so the
estimate reflects on-plane measurement noise rather than multi-plane
structure. The clip range keeps degenerate pilots from collapsing or
exploding the thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ransac import estimate_homography_ransac


@dataclass(frozen=True)
class SceneScaleEstimate:
    sigma_hat: float
    sigma_raw: float
    pilot_inliers: int
    pilot_success: bool


def estimate_scene_sigma(
    x1: np.ndarray,
    x2: np.ndarray,
    random_state: int,
    pilot_threshold: float = 5.0,
    max_iterations: int = 1500,
    min_support: int = 8,
    clip_range: tuple[float, float] = (0.6, 2.5),
) -> SceneScaleEstimate:
    """Estimate the per-scene noise scale from one loose pilot RANSAC fit."""
    result = estimate_homography_ransac(
        x1,
        x2,
        threshold=float(pilot_threshold),
        max_iterations=int(max_iterations),
        min_support=int(min_support),
        random_state=int(random_state),
        refine=True,
    )
    lo, hi = float(clip_range[0]), float(clip_range[1])
    if not result.success or result.n_inliers < min_support:
        return SceneScaleEstimate(1.0, float("nan"), int(result.n_inliers), False)
    r = result.residuals[result.inlier_mask]
    r = r[np.isfinite(r)]
    med = float(np.median(r))
    sigma_raw = float(1.4826 * np.median(np.abs(r - med)))
    sigma_hat = float(np.clip(sigma_raw, lo, hi))
    return SceneScaleEstimate(sigma_hat, sigma_raw, int(result.n_inliers), True)


def scale_config_thresholds(config: dict, sigma_hat: float) -> dict:
    """Return a deep-copied config with pixel thresholds scaled by sigma_hat.

    Scaled keys: ransac.threshold, hkm.tau_abs, hkm.sigma_min, and
    hkm.merge_threshold (all in pixels). Dimensionless settings
    (tau_norm, huber_c, energy weights) are left unchanged.
    """
    import json

    cfg = json.loads(json.dumps(config))
    s = float(sigma_hat)
    cfg.setdefault("ransac", {})["threshold"] = float(cfg["ransac"].get("threshold", 2.5)) * s
    hkm = cfg.setdefault("hkm", {})
    hkm["tau_abs"] = float(hkm.get("tau_abs", 4.0)) * s
    hkm["sigma_min"] = float(hkm.get("sigma_min", 0.75)) * s
    hkm["merge_threshold"] = float(hkm.get("merge_threshold", 5.0)) * s
    return cfg
