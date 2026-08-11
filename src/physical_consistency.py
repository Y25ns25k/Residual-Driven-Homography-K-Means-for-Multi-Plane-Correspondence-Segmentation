from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from .geometry import functional_homography_distance, symmetric_transfer_errors


@dataclass(frozen=True)
class Rank4Score:
    score: float
    trivial: bool
    n_homographies: int
    max_rank: int
    min_planes: int
    center: bool
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    tail_energy: float = 0.0
    total_energy: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "score": float(self.score),
            "trivial": bool(self.trivial),
            "n_homographies": int(self.n_homographies),
            "max_rank": int(self.max_rank),
            "min_planes": int(self.min_planes),
            "center": bool(self.center),
            "singular_values": self.singular_values.astype(float).tolist(),
            "tail_energy": float(self.tail_energy),
            "total_energy": float(self.total_energy),
        }


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    reason: str
    metrics: dict[str, object] = field(default_factory=dict)


def _as_homography_list(homographies: Iterable[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(h, dtype=np.float64).reshape(3, 3) for h in homographies]


def normalize_homographies(
    Hs: Iterable[np.ndarray],
    mode: str = "sign_aligned_frobenius",
    eps: float = 1e-12,
) -> np.ndarray:
    """Vectorize and normalize homographies for consistency diagnostics."""
    homographies = _as_homography_list(Hs)
    if not homographies:
        return np.empty((0, 9), dtype=np.float64)
    vecs = np.vstack([h.reshape(1, 9) for h in homographies]).astype(np.float64)

    if mode in {"sign_aligned_frobenius", "frobenius"}:
        norms = np.linalg.norm(vecs, axis=1)
        norms = np.where(norms > eps, norms, 1.0)
        vecs = vecs / norms[:, None]
        if mode == "sign_aligned_frobenius" and len(vecs) > 1:
            reference = vecs[0].copy()
            for i in range(1, len(vecs)):
                if float(np.dot(reference, vecs[i])) < 0.0:
                    vecs[i] *= -1.0
        return vecs

    if mode == "h33":
        scales = vecs[:, 8]
        fallback = np.linalg.norm(vecs, axis=1)
        scales = np.where(np.abs(scales) > eps, scales, np.where(fallback > eps, fallback, 1.0))
        return vecs / scales[:, None]

    raise ValueError("mode must be sign_aligned_frobenius, frobenius, or h33")


def rank4_score(
    Hs: Iterable[np.ndarray],
    max_rank: int = 4,
    min_planes: int = 5,
    center: bool = True,
    mode: str = "sign_aligned_frobenius",
    eps: float = 1e-12,
) -> Rank4Score:
    """Return relative tail energy beyond a rank-4 homography subspace.

    The score is diagnostic. It is marked trivial/non-discriminative when too
    few homographies are present to test the affine rank constraint.
    """
    homographies = _as_homography_list(Hs)
    n = len(homographies)
    if n == 0:
        return Rank4Score(0.0, True, 0, max_rank, min_planes, center)

    M = normalize_homographies(homographies, mode=mode, eps=eps)
    if center:
        M = M - M.mean(axis=0, keepdims=True)

    try:
        singular_values = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return Rank4Score(float("inf"), False, n, max_rank, min_planes, center)

    total_energy = float(np.sum(singular_values**2))
    tail_energy = float(np.sum(singular_values[max_rank:] ** 2))
    score = 0.0 if total_energy < eps else float(np.sqrt(tail_energy / total_energy))

    affine_sample_limit = max_rank + 1 if center else max_rank
    trivial = n < min_planes or n <= affine_sample_limit
    return Rank4Score(
        score=0.0 if trivial else score,
        trivial=trivial,
        n_homographies=n,
        max_rank=max_rank,
        min_planes=min_planes,
        center=center,
        singular_values=singular_values,
        tail_energy=tail_energy,
        total_energy=total_energy,
    )


def functional_distance(
    Ha: np.ndarray,
    Hb: np.ndarray,
    source_points_or_grid: np.ndarray,
) -> float:
    return functional_homography_distance(Ha, Hb, source_points_or_grid)


def physical_candidate_score(
    existing_Hs: Iterable[np.ndarray],
    candidate_H: np.ndarray,
    sample_points: Optional[np.ndarray] = None,
    max_rank: int = 4,
    min_planes: int = 5,
    center: bool = True,
) -> dict[str, object]:
    existing = _as_homography_list(existing_Hs)
    candidate = np.asarray(candidate_H, dtype=np.float64).reshape(3, 3)
    before = rank4_score(existing, max_rank=max_rank, min_planes=min_planes, center=center)
    after = rank4_score(existing + [candidate], max_rank=max_rank, min_planes=min_planes, center=center)

    duplicate_distance = float("inf")
    if sample_points is not None and len(existing) > 0:
        distances = [functional_distance(h, candidate, sample_points) for h in existing]
        duplicate_distance = float(np.min(distances)) if distances else float("inf")

    return {
        "rank4_before": before.as_dict(),
        "rank4_after": after.as_dict(),
        "rank4_delta": float(after.score - before.score),
        "min_functional_distance": duplicate_distance,
    }


def accept_candidate_by_physical_consistency(
    existing_Hs: Iterable[np.ndarray],
    candidate_H: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    candidate_inlier_mask: Optional[np.ndarray] = None,
    sample_points: Optional[np.ndarray] = None,
    min_inliers: int = 20,
    inlier_threshold: float = 1.5,
    duplicate_threshold_px: float = 5.0,
    rank4_min_planes: int = 5,
    rank4_max_rank: int = 4,
    rank4_center: bool = True,
    rank4_max_score: float = 0.35,
    rank4_max_delta: float = 0.25,
    max_median_error_factor: float = 1.25,
) -> CandidateDecision:
    """Validate a newly discovered homography without denoising the model set."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    candidate = np.asarray(candidate_H, dtype=np.float64).reshape(3, 3)

    if candidate_inlier_mask is None:
        candidate_errors = symmetric_transfer_errors(candidate, src, dst)
        inlier_mask = candidate_errors <= inlier_threshold
    else:
        inlier_mask = np.asarray(candidate_inlier_mask, dtype=bool)
        candidate_errors = symmetric_transfer_errors(candidate, src, dst)

    n_inliers = int(np.sum(inlier_mask))
    metrics: dict[str, object] = {
        "n_inliers": n_inliers,
        "inlier_threshold": float(inlier_threshold),
    }
    if n_inliers < int(min_inliers):
        return CandidateDecision(False, "too_few_inliers", metrics)

    inlier_errors = candidate_errors[inlier_mask]
    median_error = float(np.median(inlier_errors)) if len(inlier_errors) else float("inf")
    mean_error = float(np.mean(inlier_errors)) if len(inlier_errors) else float("inf")
    metrics.update({"median_error": median_error, "mean_error": mean_error})
    if median_error > float(inlier_threshold) * float(max_median_error_factor):
        return CandidateDecision(False, "insufficient_data_fit", metrics)

    score = physical_candidate_score(
        existing_Hs,
        candidate,
        sample_points=sample_points,
        max_rank=rank4_max_rank,
        min_planes=rank4_min_planes,
        center=rank4_center,
    )
    metrics.update(score)

    duplicate_distance = float(score["min_functional_distance"])
    if np.isfinite(duplicate_distance) and duplicate_distance < float(duplicate_threshold_px):
        return CandidateDecision(False, "duplicate_functional_model", metrics)

    rank_after = score["rank4_after"]
    rank_before = score["rank4_before"]
    if isinstance(rank_after, dict) and isinstance(rank_before, dict):
        after_trivial = bool(rank_after["trivial"])
        after_score = float(rank_after["score"])
        before_score = float(rank_before["score"])
        delta = after_score - before_score
        if (not after_trivial) and after_score > rank4_max_score and delta > rank4_max_delta:
            return CandidateDecision(False, "rank4_consistency_spike", metrics)

    return CandidateDecision(True, "accepted", metrics)
