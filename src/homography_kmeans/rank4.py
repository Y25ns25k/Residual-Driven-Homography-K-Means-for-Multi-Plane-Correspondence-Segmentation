"""Rank-4 homography-subspace denoising (Ke & Kanade 2003 style).

A set of plane-induced homographies between two fixed views lies, after
vectorization, in a linear subspace of dimension at most 4. Projecting the
stacked 9-vectors onto the top-4 singular subspace can therefore suppress
estimation noise — but only when K >= 5, because any K <= 4 matrices already
span a subspace of dimension <= 4 and the projection is exact (a no-op).
"""
from __future__ import annotations

import numpy as np

from .geometry import normalize_homography


def stack_homographies(H_list: list[np.ndarray]) -> np.ndarray:
    """Stack each homography as a unit-norm 9-vector into a (K, 9) matrix."""
    rows = []
    for H in H_list:
        h = np.asarray(H, dtype=np.float64).reshape(9)
        norm = np.linalg.norm(h)
        if norm < 1e-12:
            raise ValueError("homography has near-zero norm")
        rows.append(h / norm)
    return np.vstack(rows) if rows else np.empty((0, 9), dtype=np.float64)


def homography_stack_singular_values(H_list: list[np.ndarray]) -> np.ndarray:
    """Singular values of the stacked (K, 9) homography matrix."""
    M = stack_homographies(H_list)
    if M.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.svd(M, compute_uv=False)


def rank4_tail_score(
    H_list: list[np.ndarray],
    rank: int = 4,
    min_planes: int = 5,
    center: bool = True,
) -> tuple[float, bool]:
    """Relative tail energy beyond the rank-``rank`` homography subspace.

    Returns ``(score, trivial)``. ``trivial`` is True when there are too few
    homographies for the rank constraint to be discriminative (K < min_planes
    or K <= rank (+1 when centered)); the score is 0 in that case. Rows are
    sign-aligned unit 9-vectors so the score is scale/sign invariant.
    """
    K = len(H_list)
    limit = rank + 1 if center else rank
    if K == 0 or K < min_planes or K <= limit:
        return 0.0, True
    M = stack_homographies(H_list)
    for i in range(1, K):
        if float(np.dot(M[0], M[i])) < 0.0:
            M[i] *= -1.0
    if center:
        M = M - M.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(M, compute_uv=False)
    total = float(np.sum(sv**2))
    tail = float(np.sum(sv[rank:] ** 2))
    if total < 1e-12:
        return 0.0, False
    return float(np.sqrt(tail / total)), False


def rank4_candidate_consistent(
    existing: list[np.ndarray],
    candidate: np.ndarray,
    rank: int = 4,
    min_planes: int = 5,
    max_score: float = 0.35,
    max_delta: float = 0.25,
) -> tuple[bool, dict[str, float]]:
    """Physically-guided candidate gate: reject a new homography only when it
    spikes the rank-4 tail energy of an already-discriminative model set."""
    before, before_trivial = rank4_tail_score(existing, rank=rank, min_planes=min_planes)
    after, after_trivial = rank4_tail_score(existing + [np.asarray(candidate, dtype=np.float64)], rank=rank, min_planes=min_planes)
    delta = after - before
    metrics = {
        "rank4_before": before,
        "rank4_after": after,
        "rank4_delta": delta,
        "trivial": float(after_trivial),
    }
    if after_trivial:
        return True, metrics
    if after > max_score and delta > max_delta:
        return False, metrics
    return True, metrics


def denoise_homographies_rank4(
    H_list: list[np.ndarray],
    rank: int = 4,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Project stacked homographies onto their top-``rank`` singular subspace.

    Returns the denoised homographies (det-normalized, then scaled so
    H[2, 2] = 1 where possible) and a diagnostics dict with the singular
    values and whether the projection was non-trivial (K > rank).
    """
    K = len(H_list)
    info: dict[str, object] = {"K": K, "rank": int(rank), "applied": False, "singular_values": []}
    if K == 0:
        return [], info
    M = stack_homographies(H_list)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    info["singular_values"] = S.tolist()
    if K <= rank:
        # Subspace dimension <= rank already: projection is exact, keep input.
        return [np.asarray(H, dtype=np.float64).copy() for H in H_list], info
    r = int(rank)
    M_hat = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
    out: list[np.ndarray] = []
    for i in range(K):
        H = M_hat[i].reshape(3, 3)
        det = np.linalg.det(H)
        if abs(det) < 1e-12 or not np.isfinite(det):
            # Degenerate after projection: fall back to the original model.
            out.append(np.asarray(H_list[i], dtype=np.float64).copy())
            continue
        H = H / np.cbrt(det)
        try:
            out.append(normalize_homography(H))
        except ValueError:
            out.append(np.asarray(H_list[i], dtype=np.float64).copy())
    info["applied"] = True
    return out, info
