from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.geometry import apply_homography


@dataclass
class GTCorrespondenceOptions:
    points_per_plane: int = 100
    noise_sigma_px: float = 0.0
    outlier_ratio: float = 0.0
    random_seed: int = 42
    max_candidate_multiplier: int = 80


def _sample_source_points(
    source_mask: np.ndarray, plane_id: int, n_candidates: int, rng: np.random.Generator
) -> np.ndarray:
    ys, xs = np.nonzero(source_mask == plane_id)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float64)
    replace = len(xs) < n_candidates
    chosen = rng.choice(len(xs), size=n_candidates, replace=replace)
    jitter = rng.uniform(0.0, 1.0, size=(n_candidates, 2))
    return np.column_stack([xs[chosen], ys[chosen]]).astype(np.float64) + jitter


def _target_labels_at(points: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    h, w = target_mask.shape[:2]
    xy = np.rint(points).astype(np.int64)
    labels = np.full(len(points), -1, dtype=np.int32)
    valid = (
        np.isfinite(points).all(axis=1)
        & (xy[:, 0] >= 0) & (xy[:, 0] < w)
        & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    )
    labels[valid] = target_mask[xy[valid, 1], xy[valid, 0]].astype(np.int32)
    return labels


def generate_gt_correspondences(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    gt_homographies: np.ndarray,
    options: Optional[GTCorrespondenceOptions] = None,
) -> Dict[str, np.ndarray]:
    opts = options or GTCorrespondenceOptions()
    rng = np.random.default_rng(opts.random_seed)
    h, w = source_mask.shape[:2]
    plane_ids = np.arange(len(gt_homographies), dtype=np.int32)

    src_all: list[np.ndarray] = []
    dst_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    n_candidates = max(opts.points_per_plane * opts.max_candidate_multiplier, opts.points_per_plane)

    for plane_id, homography in enumerate(gt_homographies):
        src_candidates = _sample_source_points(source_mask, plane_id, n_candidates, rng)
        if len(src_candidates) == 0:
            continue
        dst_candidates = apply_homography(homography, src_candidates)
        target_labels = _target_labels_at(dst_candidates, target_mask)
        keep = target_labels == plane_id
        valid_src = src_candidates[keep]
        valid_dst = dst_candidates[keep]
        if len(valid_src) > opts.points_per_plane:
            chosen = rng.choice(len(valid_src), size=opts.points_per_plane, replace=False)
            valid_src = valid_src[chosen]
            valid_dst = valid_dst[chosen]
        src_all.append(valid_src)
        dst_all.append(valid_dst)
        labels_all.append(np.full(len(valid_src), plane_id, dtype=np.int32))

    if src_all:
        src_pts = np.vstack(src_all).astype(np.float64)
        dst_pts = np.vstack(dst_all).astype(np.float64)
        labels = np.concatenate(labels_all).astype(np.int32)
    else:
        src_pts = np.empty((0, 2), dtype=np.float64)
        dst_pts = np.empty((0, 2), dtype=np.float64)
        labels = np.empty(0, dtype=np.int32)

    if opts.noise_sigma_px > 0 and len(src_pts):
        src_pts = src_pts + rng.normal(0.0, opts.noise_sigma_px, size=src_pts.shape)
        dst_pts = dst_pts + rng.normal(0.0, opts.noise_sigma_px, size=dst_pts.shape)

    if opts.outlier_ratio > 0 and len(src_pts):
        n_outliers = int(round(opts.outlier_ratio * len(src_pts)))
        if n_outliers > 0:
            out_src = rng.uniform([0.0, 0.0], [w - 1.0, h - 1.0], size=(n_outliers, 2))
            out_dst = rng.uniform([0.0, 0.0], [w - 1.0, h - 1.0], size=(n_outliers, 2))
            src_pts = np.vstack([src_pts, out_src])
            dst_pts = np.vstack([dst_pts, out_dst])
            labels = np.concatenate([labels, np.full(n_outliers, -1, dtype=np.int32)])

    return {
        "src_pts": src_pts.astype(np.float64),
        "dst_pts": dst_pts.astype(np.float64),
        "labels": labels.astype(np.int32),
        "plane_ids": plane_ids,
    }


def generate_and_save_gt_correspondences(
    scene_dir: str | Path,
    options: Optional[GTCorrespondenceOptions] = None,
    output_name: str = "gt_correspondences_clean.npz",
) -> Dict[str, np.ndarray]:
    scene_path = Path(scene_dir)
    source_mask = np.load(scene_path / "plane_mask_source.npy")
    target_mask = np.load(scene_path / "plane_mask_target.npy")
    gt_homographies = np.load(scene_path / "gt_homographies.npy")
    data = generate_gt_correspondences(source_mask, target_mask, gt_homographies, options)
    np.savez(scene_path / output_name, **data)
    return data
