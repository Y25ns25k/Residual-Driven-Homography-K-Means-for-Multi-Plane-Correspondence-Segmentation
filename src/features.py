from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io_utils import imread


@dataclass
class FeatureMatchResult:
    src_pts: np.ndarray
    dst_pts: np.ndarray
    matches: List[cv2.DMatch]
    keypoints1: Sequence[cv2.KeyPoint]
    keypoints2: Sequence[cv2.KeyPoint]
    descriptors1: Optional[np.ndarray]
    descriptors2: Optional[np.ndarray]

    @property
    def n_matches(self) -> int:
        return len(self.matches)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape {image.shape}")


def extract_sift_features(
    image: np.ndarray,
    nfeatures: int = 8000,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Sequence[cv2.KeyPoint], Optional[np.ndarray]]:
    gray = to_gray(image)
    sift = cv2.SIFT_create(
        nfeatures=int(nfeatures),
        contrastThreshold=float(contrast_threshold),
        edgeThreshold=float(edge_threshold),
    )
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32, copy=False)
    return keypoints, descriptors


def match_descriptors(
    des1: Optional[np.ndarray],
    des2: Optional[np.ndarray],
    ratio_threshold: float = 0.75,
    checks: int = 50,
) -> List[cv2.DMatch]:
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=int(checks))
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
    good: List[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    return good


def match_images(
    image1: np.ndarray,
    image2: np.ndarray,
    nfeatures: int = 8000,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    ratio_threshold: float = 0.75,
) -> FeatureMatchResult:
    kp1, des1 = extract_sift_features(image1, nfeatures, contrast_threshold, edge_threshold)
    kp2, des2 = extract_sift_features(image2, nfeatures, contrast_threshold, edge_threshold)
    matches = match_descriptors(des1, des2, ratio_threshold)
    if matches:
        src_pts = np.float64([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float64([kp2[m.trainIdx].pt for m in matches])
    else:
        src_pts = np.empty((0, 2), dtype=np.float64)
        dst_pts = np.empty((0, 2), dtype=np.float64)
    return FeatureMatchResult(src_pts, dst_pts, matches, kp1, kp2, des1, des2)


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return imread(path, flag)
