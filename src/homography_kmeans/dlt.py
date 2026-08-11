from __future__ import annotations

import warnings

import numpy as np

from .geometry import normalize_homography, normalize_points_2d


class HomographyEstimationError(ValueError):
    """Raised when a homography cannot be estimated from the input points."""


def _as_points(x: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(x, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise HomographyEstimationError(f"{name} must have shape (N, 2)")
    if len(pts) < 4:
        raise HomographyEstimationError(f"{name} needs at least four points")
    if not np.all(np.isfinite(pts)):
        raise HomographyEstimationError(f"{name} contains non-finite values")
    return pts


def is_degenerate_configuration(x: np.ndarray, tol: float = 1e-9) -> bool:
    pts = np.asarray(x, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        return True
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered, tol=tol) < 2:
        return True
    # Reject nearly collinear minimal samples by checking triangle areas.
    max_area = 0.0
    for i in range(len(pts) - 2):
        a = pts[i + 1 :] - pts[i]
        if len(a) < 2:
            continue
        cross = np.abs(a[:-1, 0] * a[1:, 1] - a[:-1, 1] * a[1:, 0])
        if len(cross):
            max_area = max(max_area, float(np.max(cross) * 0.5))
    return max_area < tol


def _build_dlt_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    n = len(x1)
    x = x1[:, 0]
    y = x1[:, 1]
    u = x2[:, 0]
    v = x2[:, 1]
    A = np.zeros((2 * n, 9), dtype=np.float64)
    A[0::2, 3] = -x
    A[0::2, 4] = -y
    A[0::2, 5] = -1.0
    A[0::2, 6] = v * x
    A[0::2, 7] = v * y
    A[0::2, 8] = v
    A[1::2, 0] = x
    A[1::2, 1] = y
    A[1::2, 2] = 1.0
    A[1::2, 6] = -u * x
    A[1::2, 7] = -u * y
    A[1::2, 8] = -u
    return A


def estimate_homography_dlt(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    src = _as_points(x1, "x1")
    dst = _as_points(x2, "x2")
    if src.shape != dst.shape:
        raise HomographyEstimationError("x1 and x2 must have the same shape")
    if is_degenerate_configuration(src) or is_degenerate_configuration(dst):
        raise HomographyEstimationError("degenerate point configuration")

    try:
        src_n, T_src = normalize_points_2d(src)
        dst_n, T_dst = normalize_points_2d(dst)
    except ValueError as exc:
        raise HomographyEstimationError(str(exc)) from exc

    A = _build_dlt_matrix(src_n, dst_n)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(src):
            raise HomographyEstimationError("weights must have length N")
        if not np.all(np.isfinite(w)):
            raise HomographyEstimationError("weights contain non-finite values")
        w = np.clip(w, 1e-12, None)
        sw = np.sqrt(w)
        A[0::2] *= sw[:, None]
        A[1::2] *= sw[:, None]

    try:
        _, s, vt = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise HomographyEstimationError("SVD failed during DLT") from exc
    if len(s) > 1:
        cond = s[0] / max(s[-1], 1e-15)
        if cond > 1e13:
            warnings.warn(f"DLT matrix is ill-conditioned (condition number {cond:.2e})", RuntimeWarning, stacklevel=2)

    H_norm = vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    try:
        H = normalize_homography(H)
    except ValueError as exc:
        raise HomographyEstimationError(str(exc)) from exc
    if not np.all(np.isfinite(H)):
        raise HomographyEstimationError("estimated homography is non-finite")
    return H


# Compatibility with the legacy module naming.
normalized_dlt = estimate_homography_dlt
