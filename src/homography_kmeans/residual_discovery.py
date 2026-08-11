from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dlt import HomographyEstimationError, estimate_homography_dlt
from .energy import EnergyConfig, compute_energy
from .geometry import symmetric_transfer_error
from .ransac import estimate_homography_ransac


@dataclass(frozen=True)
class DiscoveryDecision:
    accepted: bool
    support: int
    delta_energy: float
    reason: str
    median_improvement: float = 0.0
    spatial_coverage: float = 0.0
    split_validated: bool = False


def _best_existing_residuals(homographies: list[np.ndarray], x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if not homographies:
        return np.full(len(x1), np.inf, dtype=np.float64)
    residuals = np.full((len(x1), len(homographies)), np.inf, dtype=np.float64)
    for k, H in enumerate(homographies):
        residuals[:, k] = symmetric_transfer_error(H, x1, x2)
    return np.min(residuals, axis=1)


def _spatial_coverage(points: np.ndarray, image_shape: tuple[int, int] | None) -> float:
    pts = np.asarray(points, dtype=np.float64)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) < 2:
        return 0.0
    extent = np.maximum(np.ptp(finite, axis=0), 0.0)
    area = float(extent[0] * extent[1])
    if image_shape is not None:
        denom = float(max(1, image_shape[0] * image_shape[1]))
    else:
        span = np.maximum(np.ptp(np.asarray(points, dtype=np.float64), axis=0), 1.0)
        denom = float(max(1e-12, span[0] * span[1]))
    return area / denom


def _median_improvement_ratio(old_residuals: np.ndarray, new_residuals: np.ndarray) -> float:
    old = np.asarray(old_residuals, dtype=np.float64)
    new = np.asarray(new_residuals, dtype=np.float64)
    finite = np.isfinite(old) & np.isfinite(new)
    if not np.any(finite):
        return 1.0
    old_med = float(np.median(old[finite]))
    new_med = float(np.median(new[finite]))
    if old_med <= 1e-12:
        return 0.0
    return float((old_med - new_med) / old_med)


def _split_validation_passes(
    homographies: list[np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
    inlier_idx: np.ndarray,
    improvement_margin: float,
    random_state: int,
) -> bool:
    if len(inlier_idx) < 8:
        return True
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(inlier_idx)
    mid = max(4, len(perm) // 2)
    fit_idx = perm[:mid]
    val_idx = perm[mid:]
    if len(val_idx) < 4:
        return True
    try:
        H = estimate_homography_dlt(x1[fit_idx], x2[fit_idx])
    except HomographyEstimationError:
        return False
    old_best = _best_existing_residuals(homographies, x1[val_idx], x2[val_idx])
    new_residuals = symmetric_transfer_error(H, x1[val_idx], x2[val_idx])
    return _median_improvement_ratio(old_best, new_residuals) >= improvement_margin


def discover_from_outliers(
    homographies: list[np.ndarray],
    labels: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float,
    max_iterations: int,
    confidence: float,
    min_support: int,
    random_state: int,
    energy_config: EnergyConfig,
    eps_energy: float,
    scale_adaptive: bool = True,
    image_shape: tuple[int, int] | None = None,
    conservative: bool = False,
    discovery_improvement_margin: float = 0.2,
    spatial_coverage_min: float = 0.05,
    split_validation: bool = False,
    local_sampling: bool = False,
    local_k: int = 50,
    from_all_points: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, DiscoveryDecision]:
    """Propose one new homography and accept it only if the global energy drops.

    By default the proposal is fitted on the current outlier pool. With
    ``from_all_points`` the proposal RANSAC runs on the full point set, which
    can recover planes whose points were absorbed by existing clusters; the
    energy gate (model penalty + global reassignment) still decides acceptance.
    """
    outlier_idx = np.flatnonzero(labels < 0)
    if not from_all_points and len(outlier_idx) < max(4, min_support):
        return homographies, labels, DiscoveryDecision(False, len(outlier_idx), 0.0, "too_few_outliers")
    cand_idx = np.arange(len(x1), dtype=np.int64) if from_all_points else outlier_idx
    if len(cand_idx) < max(4, min_support):
        return homographies, labels, DiscoveryDecision(False, len(cand_idx), 0.0, "too_few_candidates")

    old_energy, _, _, _ = compute_energy(homographies, x1, x2, config=energy_config, scale_adaptive=scale_adaptive)
    result = estimate_homography_ransac(
        x1[cand_idx],
        x2[cand_idx],
        threshold=threshold,
        max_iterations=max_iterations,
        confidence=confidence,
        min_support=min_support,
        random_state=random_state,
        refine=True,
        local_sampling=local_sampling,
        local_k=local_k,
    )
    if not result.success or result.homography is None or result.n_inliers < min_support:
        return homographies, labels, DiscoveryDecision(False, result.n_inliers, 0.0, "ransac_failed")

    global_inliers = cand_idx[result.inlier_mask]
    coverage = _spatial_coverage(x1[global_inliers], image_shape)
    improvement = 1.0
    split_ok = False
    if conservative:
        old_best = _best_existing_residuals(homographies, x1[global_inliers], x2[global_inliers])
        new_residuals = symmetric_transfer_error(result.homography, x1[global_inliers], x2[global_inliers])
        improvement = _median_improvement_ratio(old_best, new_residuals)
        if improvement < float(discovery_improvement_margin):
            return homographies, labels, DiscoveryDecision(
                False,
                result.n_inliers,
                0.0,
                "insufficient_residual_improvement",
                improvement,
                coverage,
            )
        if coverage < float(spatial_coverage_min):
            return homographies, labels, DiscoveryDecision(
                False,
                result.n_inliers,
                0.0,
                "insufficient_spatial_coverage",
                improvement,
                coverage,
            )
        split_ok = _split_validation_passes(
            homographies,
            x1,
            x2,
            global_inliers,
            float(discovery_improvement_margin),
            random_state + 171,
        )
        if split_validation and not split_ok:
            return homographies, labels, DiscoveryDecision(
                False,
                result.n_inliers,
                0.0,
                "split_validation_failed",
                improvement,
                coverage,
                split_ok,
            )

    candidate = homographies + [result.homography]
    new_energy, new_labels, _, _ = compute_energy(candidate, x1, x2, config=energy_config, scale_adaptive=scale_adaptive)
    delta = float(new_energy - old_energy)
    if delta > -float(eps_energy):
        return homographies, labels, DiscoveryDecision(
            False,
            result.n_inliers,
            delta,
            "energy_not_improved",
            improvement,
            coverage,
            split_ok,
        )
    return candidate, new_labels.astype(np.int32), DiscoveryDecision(
        True,
        result.n_inliers,
        delta,
        "accepted",
        improvement,
        coverage,
        split_ok,
    )
