from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from .geometry import apply_homography, reprojection_errors, symmetric_transfer_errors


class HomographyEstimationError(ValueError):
    """Raised when a homography cannot be estimated robustly."""


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise HomographyEstimationError(f"{name} must have shape (N, 2)")
    if len(pts) < 4:
        raise HomographyEstimationError(f"{name} needs at least 4 points")
    if not np.all(np.isfinite(pts)):
        raise HomographyEstimationError(f"{name} contains non-finite values")
    return pts


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise HomographyEstimationError("points must have shape (N, 2)")
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid[None, :]
    mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_dist < 1e-12:
        raise HomographyEstimationError("degenerate point set: zero spread")
    scale = np.sqrt(2.0) / mean_dist
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    normalized = (transform @ pts_h.T).T[:, :2]
    return normalized, transform


def is_degenerate_configuration(points: np.ndarray, tol: float = 1e-9) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        return True
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    return np.linalg.matrix_rank(centered, tol=tol) < 2


def _build_dlt_matrix(src_norm: np.ndarray, dst_norm: np.ndarray) -> np.ndarray:
    n = len(src_norm)
    x = src_norm[:, 0]
    y = src_norm[:, 1]
    u = dst_norm[:, 0]
    v = dst_norm[:, 1]
    matrix = np.zeros((2 * n, 9), dtype=np.float64)
    matrix[0::2, 3] = -x
    matrix[0::2, 4] = -y
    matrix[0::2, 5] = -1.0
    matrix[0::2, 6] = v * x
    matrix[0::2, 7] = v * y
    matrix[0::2, 8] = v
    matrix[1::2, 0] = x
    matrix[1::2, 1] = y
    matrix[1::2, 2] = 1.0
    matrix[1::2, 6] = -u * x
    matrix[1::2, 7] = -u * y
    matrix[1::2, 8] = -u
    return matrix


def normalized_dlt(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    weights: Optional[np.ndarray] = None,
    warn_condition: bool = True,
) -> np.ndarray:
    src = _as_points(src_pts, "src_pts")
    dst = _as_points(dst_pts, "dst_pts")
    if len(src) != len(dst):
        raise HomographyEstimationError("src_pts and dst_pts must have the same length")
    if is_degenerate_configuration(src) or is_degenerate_configuration(dst):
        raise HomographyEstimationError("degenerate point configuration")

    src_norm, t_src = normalize_points(src)
    dst_norm, t_dst = normalize_points(dst)
    matrix = _build_dlt_matrix(src_norm, dst_norm)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(src):
            raise HomographyEstimationError("weights must have length N")
        if not np.all(np.isfinite(w)):
            raise HomographyEstimationError("weights contain non-finite values")
        w = np.clip(w, 1e-12, None)
        matrix[0::2] *= np.sqrt(w)[:, None]
        matrix[1::2] *= np.sqrt(w)[:, None]

    try:
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise HomographyEstimationError("SVD failed during DLT") from exc

    if warn_condition and len(singular_values) > 1:
        cond = singular_values[0] / max(singular_values[-1], 1e-15)
        if cond > 1e12:
            warnings.warn(
                f"DLT matrix is ill-conditioned (condition number {cond:.2e})",
                RuntimeWarning,
                stacklevel=2,
            )

    h_norm = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(t_dst) @ h_norm @ t_src
    scale = homography[2, 2]
    if abs(scale) < 1e-12:
        norm = np.linalg.norm(homography)
        if norm < 1e-12:
            raise HomographyEstimationError("estimated homography has near-zero scale")
        homography /= norm
    else:
        homography /= scale
    if not np.all(np.isfinite(homography)):
        raise HomographyEstimationError("estimated homography is non-finite")
    return homography
