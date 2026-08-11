"""Residual-driven cluster splitting — the dual of merging.

Sequential RANSAC initialization sometimes covers two real planes with one
homography (initialization undersegmentation); outlier-pool discovery cannot
fix this because the affected points are *inliers* of the merged model. This
operator targets the cluster with the largest robust residual scale, refits
two homographies to its members, and accepts the split only if the global
energy (which already charges lambda_K per model) strictly decreases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .energy import EnergyConfig, compute_energy
from .ransac import estimate_homography_ransac


@dataclass(frozen=True)
class SplitDecision:
    accepted: bool
    cluster: int
    support_a: int
    support_b: int
    delta_energy: float
    reason: str


def _cluster_scales(homographies, labels, x1, x2, sigma_min):
    from .energy import estimate_scales

    return estimate_scales(homographies, labels, x1, x2, sigma_min)


def split_worst_cluster(
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
) -> tuple[list[np.ndarray], np.ndarray, SplitDecision]:
    if not homographies:
        return homographies, labels, SplitDecision(False, -1, 0, 0, 0.0, "no_models")

    scales = _cluster_scales(homographies, labels, x1, x2, energy_config.sigma_min)
    counts = np.array([int(np.sum(labels == k)) for k in range(len(homographies))])
    eligible = np.flatnonzero(counts >= 2 * max(4, min_support))
    if len(eligible) == 0:
        return homographies, labels, SplitDecision(False, -1, 0, 0, 0.0, "no_eligible_cluster")
    k = int(eligible[np.argmax(scales[eligible])])
    member_idx = np.flatnonzero(labels == k)

    result_a = estimate_homography_ransac(
        x1[member_idx],
        x2[member_idx],
        threshold=threshold,
        max_iterations=max_iterations,
        confidence=confidence,
        min_support=min_support,
        random_state=random_state,
        refine=True,
    )
    if not result_a.success or result_a.homography is None or result_a.n_inliers < min_support:
        return homographies, labels, SplitDecision(False, k, result_a.n_inliers, 0, 0.0, "ransac_a_failed")
    rest = member_idx[~result_a.inlier_mask]
    if len(rest) < max(4, min_support):
        return homographies, labels, SplitDecision(False, k, result_a.n_inliers, len(rest), 0.0, "remainder_too_small")
    result_b = estimate_homography_ransac(
        x1[rest],
        x2[rest],
        threshold=threshold,
        max_iterations=max_iterations,
        confidence=confidence,
        min_support=min_support,
        random_state=random_state + 313,
        refine=True,
    )
    if not result_b.success or result_b.homography is None or result_b.n_inliers < min_support:
        return homographies, labels, SplitDecision(False, k, result_a.n_inliers, result_b.n_inliers, 0.0, "ransac_b_failed")

    old_energy, _, _, _ = compute_energy(homographies, x1, x2, config=energy_config, scale_adaptive=scale_adaptive)
    candidate = [H for j, H in enumerate(homographies) if j != k] + [result_a.homography, result_b.homography]
    new_energy, new_labels, _, _ = compute_energy(candidate, x1, x2, config=energy_config, scale_adaptive=scale_adaptive)
    delta = float(new_energy - old_energy)
    if delta > -float(eps_energy):
        return homographies, labels, SplitDecision(
            False, k, result_a.n_inliers, result_b.n_inliers, delta, "energy_not_improved"
        )
    return candidate, new_labels.astype(np.int32), SplitDecision(
        True, k, result_a.n_inliers, result_b.n_inliers, delta, "accepted"
    )
