from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    x1: np.ndarray
    x2: np.ndarray
    method: str
    n_keypoints1: int
    n_keypoints2: int


def match_image_pair(image1: np.ndarray, image2: np.ndarray, ratio: float = 0.75, nfeatures: int = 4000) -> MatchResult:
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if image1.ndim == 3 else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if image2.ndim == 3 else image2
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=nfeatures)
        norm = cv2.NORM_L2
        method = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=nfeatures)
        norm = cv2.NORM_HAMMING
        method = "ORB"
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return MatchResult(np.empty((0, 2)), np.empty((0, 2)), method, len(kp1), len(kp2))
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(des1, des2, k=2)
    src: list[tuple[float, float]] = []
    dst: list[tuple[float, float]] = []
    for item in pairs:
        if len(item) < 2:
            continue
        a, b = item
        if a.distance < ratio * b.distance:
            src.append(kp1[a.queryIdx].pt)
            dst.append(kp2[a.trainIdx].pt)
    return MatchResult(np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64), method, len(kp1), len(kp2))
