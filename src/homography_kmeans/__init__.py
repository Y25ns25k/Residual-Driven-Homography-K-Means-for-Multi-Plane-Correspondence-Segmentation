"""Residual-driven homography K-Means package.

The package intentionally keeps the main method classical and interpretable:
Sequential RANSAC proposes greedy initial homographies, then a scale-adaptive
assignment/refit loop revisits all correspondences and optionally discovers or
merges models using residual/energy criteria.
"""

from .dlt import HomographyEstimationError, estimate_homography_dlt
from .hkm import FitResult, ResidualHomographyKMeans
from .ransac import RansacResult, estimate_homography_ransac
from .sequential import SequentialRansacResult, sequential_ransac

# Compatibility names for the legacy flat project imports.
HomographyKMeans = ResidualHomographyKMeans
KMeansResult = FitResult

__all__ = [
    "FitResult",
    "HomographyEstimationError",
    "HomographyKMeans",
    "KMeansResult",
    "RansacResult",
    "ResidualHomographyKMeans",
    "SequentialRansacResult",
    "estimate_homography_dlt",
    "estimate_homography_ransac",
    "sequential_ransac",
]
