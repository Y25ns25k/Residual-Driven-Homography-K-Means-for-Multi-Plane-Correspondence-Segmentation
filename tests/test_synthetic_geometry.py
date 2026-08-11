"""Test that GT homographies are physically consistent with 3D projection.

H = K (R - t n^T / d) K^{-1} must map source pixels to target pixels with < 0.1px error.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.geometry import (
    apply_homography,
    plane_induced_homography,
    project_points,
    rotation_matrix_from_euler,
    transform_points,
)


def _make_scene(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    focal = 300.0
    H, W = 600, 800
    K = np.array([[focal, 0, (W - 1) / 2], [0, focal, (H - 1) / 2], [0, 0, 1.0]])
    yaw = rng.uniform(-8, 8)
    pitch = rng.uniform(-5, 5)
    roll = rng.uniform(-2, 2)
    R = rotation_matrix_from_euler(pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)
    t = rng.uniform(-0.2, 0.2, size=3)
    # Front wall plane: z = 3, normal = (0,0,-1), origin = (0,0,3)
    origin = np.array([0.0, 0.0, 3.0])
    normal = np.array([0.0, 0.0, -1.0])
    d = -float(normal @ origin)  # = -3*(-1) = 3
    return {"K": K, "R": R, "t": t, "origin": origin, "normal": normal, "d": d}


def test_plane_homography_accuracy():
    scene = _make_scene(42)
    K, R, t, origin, normal, d = (
        scene["K"], scene["R"], scene["t"],
        scene["origin"], scene["normal"], scene["d"],
    )
    H = plane_induced_homography(K, R, t, normal, d)
    assert H.shape == (3, 3)
    # Sample points on the plane, project to view 1, apply H, compare to view 2
    rng = np.random.default_rng(0)
    us = rng.uniform(-0.9, 0.9, 50)
    vs = rng.uniform(-0.5, 0.5, 50)
    world_pts = origin[None, :] + us[:, None] * np.array([1, 0, 0]) + vs[:, None] * np.array([0, 1, 0])
    src_2d, src_depth = project_points(K, world_pts)
    valid = np.isfinite(src_2d).all(axis=1) & (src_depth > 0)
    assert np.sum(valid) >= 20
    src_valid = src_2d[valid]
    world_valid = world_pts[valid]
    target_pts = transform_points(world_valid, R, t)
    dst_2d, dst_depth = project_points(K, target_pts)
    valid2 = np.isfinite(dst_2d).all(axis=1) & (dst_depth > 0)
    assert np.sum(valid2) >= 10
    warped = apply_homography(H, src_valid[valid2])
    errors = np.linalg.norm(warped - dst_2d[valid2], axis=1)
    assert np.nanmean(errors) < 0.05, f"GT H error too large: {np.nanmean(errors):.4f}px"


def test_homography_normalization():
    scene = _make_scene(7)
    K, R, t, origin, normal, d = (
        scene["K"], scene["R"], scene["t"],
        scene["origin"], scene["normal"], scene["d"],
    )
    H = plane_induced_homography(K, R, t, normal, d)
    assert abs(H[2, 2] - 1.0) < 1e-9


def test_offset_sign_convention():
    """n^T * X + d = 0 must hold for origin: d = -n^T * origin."""
    origin = np.array([0.0, 0.0, 3.0])
    normal = np.array([0.0, 0.0, -1.0])
    d = -float(normal @ origin)
    assert abs(float(normal @ origin) + d) < 1e-12
