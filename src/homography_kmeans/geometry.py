from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def normalize_homography(H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(H, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    out = h.copy()
    if abs(out[2, 2]) > eps:
        return out / out[2, 2]
    norm = np.linalg.norm(out)
    if norm < eps:
        raise ValueError("homography has near-zero norm")
    return out / norm


def to_homogeneous(x: np.ndarray) -> np.ndarray:
    pts = np.asarray(x, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("x must be a 2D array")
    return np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])


def from_homogeneous(xh: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pts = np.asarray(xh, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("xh must have shape (N, D+1)")
    denom = pts[:, -1]
    out = np.full((len(pts), pts.shape[1] - 1), np.nan, dtype=np.float64)
    valid = np.abs(denom) > eps
    out[valid] = pts[valid, :-1] / denom[valid, None]
    return out


def normalize_points_2d(x: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(x, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError("x must have shape (N, 2)")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain non-finite values")
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid[None, :]
    mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_dist < eps:
        raise ValueError("degenerate point set: zero spread")
    scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    x_norm = from_homogeneous((T @ to_homogeneous(pts).T).T)
    return x_norm, T


def apply_homography(H: np.ndarray, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(H, dtype=np.float64)
    pts = np.asarray(x, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("x must have shape (N, 2)")
    return from_homogeneous((h @ to_homogeneous(pts).T).T, eps=eps)


def _forward_backward_distances(H: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(x1, dtype=np.float64)
    dst = np.asarray(x2, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
        raise ValueError("x1 and x2 must both have shape (N, 2)")
    warped = apply_homography(H, src)
    forward = np.linalg.norm(warped - dst, axis=1)
    try:
        inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        inf = np.full(len(src), np.inf, dtype=np.float64)
        return inf, inf
    back = apply_homography(inv, dst)
    backward = np.linalg.norm(back - src, axis=1)
    forward[~np.isfinite(forward)] = np.inf
    backward[~np.isfinite(backward)] = np.inf
    return forward, backward


def symmetric_transfer_error(H: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Non-squared symmetric transfer error in pixels.

    The default follows the project spec:
    ``0.5 * (||pi(H x) - x'||_2 + ||pi(H^-1 x') - x||_2)``.
    """

    forward, backward = _forward_backward_distances(H, x1, x2)
    return 0.5 * (forward + backward)


def symmetric_transfer_error_squared(H: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    forward, backward = _forward_backward_distances(H, x1, x2)
    return 0.5 * (forward**2 + backward**2)


def grid_points(image_shape: Tuple[int, int], nx: int = 7, ny: int = 5) -> np.ndarray:
    h, w = image_shape[:2]
    xs = np.linspace(0, w - 1, nx)
    ys = np.linspace(0, h - 1, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def image_corners(image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    return np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])


def functional_warp_distance(
    H_a: np.ndarray,
    H_b: np.ndarray,
    image_shape: Tuple[int, int] | None = None,
    sample_points: np.ndarray | None = None,
    normalize_by_diagonal: bool = False,
) -> float:
    if sample_points is None:
        if image_shape is None:
            raise ValueError("image_shape or sample_points must be provided")
        pts = grid_points(image_shape)
    else:
        pts = np.asarray(sample_points, dtype=np.float64)
    a = apply_homography(H_a, pts)
    b = apply_homography(H_b, pts)
    d = np.linalg.norm(a - b, axis=1)
    finite = np.isfinite(d)
    if not np.any(finite):
        return float("inf")
    value = float(np.median(d[finite]))
    if normalize_by_diagonal:
        if image_shape is None:
            span = np.ptp(pts, axis=0)
            diag = float(np.linalg.norm(span))
        else:
            diag = float(np.linalg.norm([image_shape[1], image_shape[0]]))
        value /= max(diag, 1e-12)
    return value


def plane_induced_homography(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    normal: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Return ``K (R - t n^T / d) K^-1`` for controlled synthetic scenes."""

    k = np.asarray(K, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    tv = np.asarray(t, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    d = float(distance)
    if abs(d) < 1e-12:
        raise ValueError("plane distance must be non-zero")
    return normalize_homography(k @ (r - np.outer(tv, n) / d) @ np.linalg.inv(k))
