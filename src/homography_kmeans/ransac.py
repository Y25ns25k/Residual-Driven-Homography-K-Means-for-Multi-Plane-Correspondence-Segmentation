from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dlt import HomographyEstimationError, estimate_homography_dlt, is_degenerate_configuration
from .geometry import symmetric_transfer_error


@dataclass
class RansacResult:
    homography: np.ndarray | None
    inlier_mask: np.ndarray
    residuals: np.ndarray
    n_iter: int
    success: bool
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def inliers(self) -> np.ndarray:
        return np.flatnonzero(self.inlier_mask)

    @property
    def n_inliers(self) -> int:
        return int(np.sum(self.inlier_mask))


def _empty_result(n: int) -> RansacResult:
    return RansacResult(None, np.zeros(n, dtype=bool), np.full(n, np.inf), 0, False, {})


def _adaptive_iteration_limit(confidence: float, inlier_ratio: float, sample_size: int) -> int:
    ratio = float(np.clip(inlier_ratio, 1e-12, 1.0 - 1e-12))
    p_good = ratio**sample_size
    denom = math.log(max(1.0 - p_good, 1e-12))
    return int(math.ceil(math.log(max(1.0 - confidence, 1e-12)) / denom))


def estimate_homography_ransac(
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float = 3.0,
    max_iterations: int = 2000,
    confidence: float = 0.999,
    min_support: int = 4,
    random_state: int | None = 42,
    refine: bool = True,
    local_sampling: bool = False,
    local_k: int = 50,
) -> RansacResult:
    """Robust homography fit; ``local_sampling`` draws the minimal sample from
    a random seed point plus three of its ``local_k`` spatial neighbors, which
    raises the per-sample inlier probability when the structure is spatially
    coherent but globally rare (low global inlier ratio)."""
    src = np.asarray(x1, dtype=np.float64)
    dst = np.asarray(x2, dtype=np.float64)
    n = len(src)
    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape or n < 4:
        return _empty_result(n)

    rng = np.random.default_rng(random_state)
    sample_size = 4
    neighbor_idx: np.ndarray | None = None
    if local_sampling and n > sample_size + 1:
        from scipy.spatial import cKDTree

        kk = min(int(local_k), n - 1)
        if kk >= sample_size - 1:
            _, nn = cKDTree(src).query(src, k=kk + 1)
            neighbor_idx = np.asarray(nn[:, 1:], dtype=np.int64)
    iteration_limit = int(max_iterations)
    best_H: np.ndarray | None = None
    best_mask = np.zeros(n, dtype=bool)
    best_residuals = np.full(n, np.inf, dtype=np.float64)
    best_median = float("inf")
    iterations_done = 0
    rejected_degenerate = 0
    rejected_estimation = 0

    for iteration in range(int(max_iterations)):
        if iteration >= iteration_limit:
            break
        iterations_done = iteration + 1
        if neighbor_idx is not None:
            seed_pt = int(rng.integers(n))
            sample = np.concatenate(
                [[seed_pt], rng.choice(neighbor_idx[seed_pt], size=sample_size - 1, replace=False)]
            )
        else:
            sample = rng.choice(n, size=sample_size, replace=False)
        if is_degenerate_configuration(src[sample]) or is_degenerate_configuration(dst[sample]):
            rejected_degenerate += 1
            continue
        try:
            H = estimate_homography_dlt(src[sample], dst[sample])
        except HomographyEstimationError:
            rejected_estimation += 1
            continue
        residuals = symmetric_transfer_error(H, src, dst)
        mask = residuals <= threshold
        support = int(np.sum(mask))
        if support < min_support:
            continue
        median = float(np.median(residuals[mask])) if support else float("inf")
        best_support = int(np.sum(best_mask))
        if support > best_support or (support == best_support and median < best_median):
            best_H = H
            best_mask = mask
            best_residuals = residuals
            best_median = median
            limit = _adaptive_iteration_limit(confidence, support / max(n, 1), sample_size)
            iteration_limit = min(iteration_limit, max(limit, iteration + 1))

    if best_H is None or int(np.sum(best_mask)) < min_support:
        return RansacResult(
            None,
            best_mask,
            best_residuals,
            iterations_done,
            False,
            {"rejected_degenerate": float(rejected_degenerate), "rejected_estimation": float(rejected_estimation)},
        )

    if refine and int(np.sum(best_mask)) >= 4:
        try:
            refined = estimate_homography_dlt(src[best_mask], dst[best_mask])
            refined_residuals = symmetric_transfer_error(refined, src, dst)
            refined_mask = refined_residuals <= threshold
            if int(np.sum(refined_mask)) >= int(np.sum(best_mask)):
                best_H = refined
                best_mask = refined_mask
                best_residuals = refined_residuals
                best_median = float(np.median(refined_residuals[refined_mask]))
        except HomographyEstimationError:
            pass

    return RansacResult(
        best_H,
        best_mask,
        best_residuals,
        iterations_done,
        True,
        {
            "median_residual": best_median,
            "rejected_degenerate": float(rejected_degenerate),
            "rejected_estimation": float(rejected_estimation),
        },
    )


# Compatibility with the legacy flat project signature.
def estimate_homography_ransac_legacy(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 3.0,
    max_iter: int = 2500,
    confidence: float = 0.999,
    min_inliers: int = 4,
    random_state: int | None = 42,
    refine: bool = True,
    error_metric: str = "symmetric",
) -> RansacResult:
    _ = error_metric
    return estimate_homography_ransac(
        src_pts,
        dst_pts,
        threshold=threshold,
        max_iterations=max_iter,
        confidence=confidence,
        min_support=min_inliers,
        random_state=random_state,
        refine=refine,
    )
