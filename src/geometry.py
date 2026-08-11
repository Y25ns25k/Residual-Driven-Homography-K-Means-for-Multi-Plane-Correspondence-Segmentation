from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def normalize_homography(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    out = h.copy()
    if abs(out[2, 2]) > eps:
        out /= out[2, 2]
    else:
        norm = np.linalg.norm(out)
        if norm < eps:
            raise ValueError("homography has near-zero norm")
        out /= norm
    return out


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array")
    return np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])


def from_homogeneous(points_h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pts = np.asarray(points_h, dtype=np.float64)
    denom = pts[:, -1]
    out = np.full((len(pts), pts.shape[1] - 1), np.nan, dtype=np.float64)
    valid = np.abs(denom) > eps
    out[valid] = pts[valid, :-1] / denom[valid, None]
    return out


def apply_homography(homography: np.ndarray, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(homography, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    warped_h = (h @ to_homogeneous(pts).T).T
    return from_homogeneous(warped_h, eps=eps)


def project_points(intrinsics: np.ndarray, points_camera: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    k = np.asarray(intrinsics, dtype=np.float64)
    pts = np.asarray(points_camera, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_camera must have shape (N, 3)")
    depth = pts[:, 2]
    proj_h = (k @ pts.T).T
    xy = from_homogeneous(proj_h, eps=eps)
    xy[depth <= eps] = np.nan
    return xy, depth


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if r.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    return (r @ pts.T).T + t[None, :]


def rotation_matrix_from_euler(
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    roll = math.radians(roll_deg)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cz, sz = math.cos(roll), math.sin(roll)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def plane_induced_homography(
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray:
    """H = K (R - t n^T / d) K^{-1}  mapping view-1 pixels to view-2 pixels.

    Convention: View-1 = K[I|0], View-2 = K[R|t].
    Plane in view-1 coords: n^T X + d = 0, with d = -n^T * origin.
    """
    k = np.asarray(intrinsics, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    d = float(plane_offset)
    if k.shape != (3, 3) or r.shape != (3, 3):
        raise ValueError("intrinsics and rotation must be 3x3")
    if abs(d) < 1e-12:
        raise ValueError("plane_offset must be non-zero")
    h = k @ (r - np.outer(t, n) / d) @ np.linalg.inv(k)
    return normalize_homography(h)


def reprojection_errors(homography: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    warped = apply_homography(homography, src)
    errors = np.linalg.norm(warped - dst, axis=1)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def symmetric_transfer_errors(homography: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    forward = reprojection_errors(homography, src_pts, dst_pts)
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return np.full(len(forward), np.inf, dtype=np.float64)
    backward = reprojection_errors(inverse, dst_pts, src_pts)
    errors = np.sqrt(0.5 * (forward ** 2 + backward ** 2))
    errors[~np.isfinite(errors)] = np.inf
    return errors


def functional_homography_distance(
    h1: np.ndarray,
    h2: np.ndarray,
    sample_points: np.ndarray,
) -> float:
    p1 = apply_homography(h1, sample_points)
    p2 = apply_homography(h2, sample_points)
    distances = np.linalg.norm(p1 - p2, axis=1)
    finite = np.isfinite(distances)
    if not np.any(finite):
        return float("inf")
    return float(np.mean(distances[finite]))


def grid_points_for_shape(image_shape: Tuple[int, int], nx: int = 6, ny: int = 5) -> np.ndarray:
    h, w = image_shape[:2]
    xs = np.linspace(0, w - 1, nx)
    ys = np.linspace(0, h - 1, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def corner_points(image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
