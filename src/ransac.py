from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .geometry import reprojection_errors, symmetric_transfer_errors
from .normalized_dlt import HomographyEstimationError, is_degenerate_configuration, normalized_dlt


@dataclass
class RansacResult:
    homography: Optional[np.ndarray]
    inlier_mask: np.ndarray
    errors: np.ndarray
    n_iter: int
    success: bool
    score: float

    @property
    def inliers(self) -> np.ndarray:
        return np.flatnonzero(self.inlier_mask)

    @property
    def n_inliers(self) -> int:
        return int(np.sum(self.inlier_mask))


@dataclass
class SequentialRansacResult:
    homographies: List[np.ndarray]
    labels: np.ndarray
    inlier_masks: List[np.ndarray]
    remaining: np.ndarray
    results: List[RansacResult]


def _empty_result(n: int) -> RansacResult:
    return RansacResult(None, np.zeros(n, dtype=bool), np.full(n, np.inf), 0, False, np.inf)


def _adaptive_iteration_limit(confidence: float, inlier_ratio: float, sample_size: int) -> int:
    inlier_ratio = float(np.clip(inlier_ratio, 1e-12, 1.0 - 1e-12))
    p_good_sample = inlier_ratio ** sample_size
    denom = math.log(max(1.0 - p_good_sample, 1e-12))
    return int(math.ceil(math.log(max(1.0 - confidence, 1e-12)) / denom))


def _errors(homography: np.ndarray, src: np.ndarray, dst: np.ndarray, metric: str) -> np.ndarray:
    if metric == "reprojection":
        return reprojection_errors(homography, src, dst)
    if metric == "symmetric":
        return symmetric_transfer_errors(homography, src, dst)
    raise ValueError("metric must be 'reprojection' or 'symmetric'")


def estimate_homography_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 3.0,
    max_iter: int = 2500,
    confidence: float = 0.999,
    min_inliers: int = 4,
    random_state: Optional[int] = 42,
    refine: bool = True,
    error_metric: str = "symmetric",
) -> RansacResult:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = len(src)
    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape or n < 4:
        return _empty_result(n)

    rng = np.random.default_rng(random_state)
    best_h: Optional[np.ndarray] = None
    best_mask = np.zeros(n, dtype=bool)
    best_errors = np.full(n, np.inf, dtype=np.float64)
    best_score = np.inf
    sample_size = 4
    iteration_limit = max_iter
    iterations_done = 0

    for iteration in range(max_iter):
        if iteration >= iteration_limit:
            break
        iterations_done = iteration + 1
        sample = rng.choice(n, size=sample_size, replace=False)
        if is_degenerate_configuration(src[sample]) or is_degenerate_configuration(dst[sample]):
            continue
        try:
            candidate = normalized_dlt(src[sample], dst[sample], warn_condition=False)
        except HomographyEstimationError:
            continue
        errors = _errors(candidate, src, dst, error_metric)
        mask = errors <= threshold
        inliers = int(np.sum(mask))
        if inliers < min_inliers:
            continue
        score = float(np.sum(np.minimum(errors, threshold) ** 2))
        if inliers > int(np.sum(best_mask)) or (inliers == int(np.sum(best_mask)) and score < best_score):
            best_h = candidate
            best_mask = mask
            best_errors = errors
            best_score = score
            limit = _adaptive_iteration_limit(confidence, inliers / n, sample_size)
            iteration_limit = min(iteration_limit, max(limit, iteration + 1))

    if best_h is None or int(np.sum(best_mask)) < min_inliers:
        return RansacResult(None, best_mask, best_errors, iterations_done, False, best_score)

    if refine and int(np.sum(best_mask)) >= 4:
        try:
            refined = normalized_dlt(src[best_mask], dst[best_mask], warn_condition=False)
            refined_errors = _errors(refined, src, dst, error_metric)
            refined_mask = refined_errors <= threshold
            refined_score = float(np.sum(np.minimum(refined_errors, threshold) ** 2))
            if int(np.sum(refined_mask)) >= int(np.sum(best_mask)) or refined_score < best_score:
                best_h = refined
                best_errors = refined_errors
                best_mask = refined_mask
                best_score = refined_score
        except HomographyEstimationError:
            pass

    return RansacResult(best_h, best_mask, best_errors, iterations_done, True, best_score)


def sequential_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 1.0,
    max_iter: int = 2500,
    confidence: float = 0.999,
    min_inliers: int = 20,
    stop_fraction: float = 0.05,
    max_planes: Optional[int] = None,
    random_state: Optional[int] = 42,
    error_metric: str = "symmetric",
) -> SequentialRansacResult:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = len(src)
    labels = np.full(n, -1, dtype=np.int32)
    remaining = np.arange(n, dtype=np.int64)
    homographies: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    results: List[RansacResult] = []
    rng = np.random.default_rng(random_state)
    stop_count = max(int(math.ceil(stop_fraction * n)), int(min_inliers))

    while len(remaining) >= stop_count and len(remaining) >= 4:
        if max_planes is not None and len(homographies) >= max_planes:
            break
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        result = estimate_homography_ransac(
            src[remaining],
            dst[remaining],
            threshold=threshold,
            max_iter=max_iter,
            confidence=confidence,
            min_inliers=max(4, min_inliers),
            random_state=seed,
            error_metric=error_metric,
        )
        if not result.success or result.n_inliers < min_inliers or result.homography is None:
            break
        global_mask = np.zeros(n, dtype=bool)
        global_inliers = remaining[result.inlier_mask]
        global_mask[global_inliers] = True
        plane_id = len(homographies)
        labels[global_inliers] = plane_id
        homographies.append(result.homography)
        masks.append(global_mask)
        results.append(result)
        remaining = remaining[~result.inlier_mask]

    return SequentialRansacResult(homographies, labels, masks, remaining, results)


def cv2_ransac(src_pts: np.ndarray, dst_pts: np.ndarray, threshold: float = 5.0) -> RansacResult:
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    n = len(src)
    if n < 4:
        return _empty_result(n)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    if homography is None or mask is None:
        return _empty_result(n)
    inlier_mask = mask.reshape(-1).astype(bool)
    errors = symmetric_transfer_errors(homography, src.astype(np.float64), dst.astype(np.float64))
    score = float(np.sum(np.minimum(errors, threshold) ** 2))
    return RansacResult(homography.astype(np.float64), inlier_mask, errors, 0, True, score)
