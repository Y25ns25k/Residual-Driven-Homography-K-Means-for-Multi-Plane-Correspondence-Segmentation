from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.features import match_images
from src.io_utils import imread


@dataclass
class SIFTCorrespondenceResult:
    src_pts: np.ndarray
    dst_pts: np.ndarray
    gt_labels: np.ndarray        # -1 if unlabeled or source_only
    n_matches: int


def _labels_at(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    xy = np.rint(points).astype(np.int64)
    labels = np.full(len(points), -1, dtype=np.int32)
    valid = (
        np.isfinite(points).all(axis=1)
        & (xy[:, 0] >= 0) & (xy[:, 0] < w)
        & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    )
    labels[valid] = mask[xy[valid, 1], xy[valid, 0]].astype(np.int32)
    return labels


def load_sift_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    nfeatures: int = 8000,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    ratio_threshold: float = 0.75,
) -> SIFTCorrespondenceResult:
    result = match_images(
        source, target,
        nfeatures=nfeatures,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        ratio_threshold=ratio_threshold,
    )
    src_pts = result.src_pts
    dst_pts = result.dst_pts
    gt_labels = np.full(len(src_pts), -1, dtype=np.int32)
    if source_mask is not None and len(src_pts):
        gt_labels = _labels_at(src_pts, source_mask)
    return SIFTCorrespondenceResult(src_pts, dst_pts, gt_labels, result.n_matches)


def load_sift_correspondences_from_scene(
    scene_dir: str | Path,
    config=None,
) -> SIFTCorrespondenceResult:
    scene_path = Path(scene_dir)
    source = imread(scene_path / "source.png", cv2.IMREAD_COLOR)
    target = imread(scene_path / "target.png", cv2.IMREAD_COLOR)
    source_mask = np.load(scene_path / "plane_mask_source.npy")
    nfeatures = 8000
    contrast_threshold = 0.04
    edge_threshold = 10.0
    ratio_threshold = 0.75
    if config is not None:
        nfeatures = int(getattr(config, "sift_nfeatures", nfeatures))
        contrast_threshold = float(getattr(config, "sift_contrast_threshold", contrast_threshold))
        edge_threshold = float(getattr(config, "sift_edge_threshold", edge_threshold))
        ratio_threshold = float(getattr(config, "ratio_threshold", ratio_threshold))
    return load_sift_correspondences(
        source, target, source_mask,
        nfeatures=nfeatures,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        ratio_threshold=ratio_threshold,
    )
