from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ransac import RansacResult, estimate_homography_ransac


@dataclass
class SequentialRansacResult:
    homographies: list[np.ndarray]
    labels: np.ndarray
    residuals: np.ndarray
    inlier_masks: list[np.ndarray]
    remaining: np.ndarray
    results: list[RansacResult]
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return len(self.homographies)


def sequential_ransac(
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float = 3.0,
    max_iterations: int = 2000,
    confidence: float = 0.999,
    min_support: int = 20,
    max_models: int | None = None,
    random_state: int | None = 42,
    stop_fraction: float = 0.0,
    **legacy_kwargs,
) -> SequentialRansacResult:
    if "max_iter" in legacy_kwargs:
        max_iterations = int(legacy_kwargs["max_iter"])
    if "min_inliers" in legacy_kwargs:
        min_support = int(legacy_kwargs["min_inliers"])
    if "max_planes" in legacy_kwargs:
        max_models = legacy_kwargs["max_planes"]

    src = np.asarray(x1, dtype=np.float64)
    dst = np.asarray(x2, dtype=np.float64)
    n = len(src)
    labels = np.full(n, -1, dtype=np.int32)
    residuals = np.full(n, np.inf, dtype=np.float64)
    remaining = np.arange(n, dtype=np.int64)
    homographies: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    results: list[RansacResult] = []
    rng = np.random.default_rng(random_state)
    stop_count = max(int(math.ceil(stop_fraction * n)), int(min_support), 4)

    while len(remaining) >= stop_count:
        if max_models is not None and len(homographies) >= int(max_models):
            break
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        result = estimate_homography_ransac(
            src[remaining],
            dst[remaining],
            threshold=threshold,
            max_iterations=max_iterations,
            confidence=confidence,
            min_support=max(4, min_support),
            random_state=seed,
            refine=True,
        )
        if not result.success or result.homography is None or result.n_inliers < min_support:
            break
        global_inliers = remaining[result.inlier_mask]
        model_id = len(homographies)
        labels[global_inliers] = model_id
        residuals[remaining] = np.minimum(residuals[remaining], result.residuals)
        global_mask = np.zeros(n, dtype=bool)
        global_mask[global_inliers] = True
        homographies.append(result.homography)
        masks.append(global_mask)
        results.append(result)
        remaining = remaining[~result.inlier_mask]

    return SequentialRansacResult(
        homographies,
        labels,
        residuals,
        masks,
        remaining,
        results,
        {"threshold": float(threshold), "min_support": float(min_support)},
    )
