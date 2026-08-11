import numpy as np

from src.homography_kmeans.geometry import apply_homography
from src.homography_kmeans.official_init import (
    homography_norm_to_pixel,
    normalization_matrix,
    normalized_to_pixel_points,
    pixel_to_normalized_points,
)


def test_consac_point_normalization_roundtrip():
    shape = (480, 640, 3)
    pts = np.array([[0.0, 0.0], [320.0, 240.0], [640.0, 480.0], [123.5, 77.0]])
    norm = pixel_to_normalized_points(pts, shape)
    back = normalized_to_pixel_points(norm, shape)
    assert np.allclose(back, pts)


def test_consac_homography_norm_to_pixel_matches_coordinate_change():
    img1_shape = (480, 640, 3)
    img2_shape = (720, 960, 3)
    H_pix_true = np.array([[1.03, 0.02, 18.0], [-0.01, 0.97, 24.0], [1e-5, -2e-5, 1.0]])
    A1 = normalization_matrix(img1_shape)
    A2 = normalization_matrix(img2_shape)
    H_norm = A2 @ H_pix_true @ np.linalg.inv(A1)
    H_pix = homography_norm_to_pixel(H_norm, img1_shape, img2_shape)
    H_pix_true = H_pix_true / H_pix_true[2, 2]
    assert np.allclose(H_pix, H_pix_true, atol=1e-8)

    pts_pix = np.array([[20.0, 30.0], [300.0, 200.0], [610.0, 420.0]])
    pts_norm = pixel_to_normalized_points(pts_pix, img1_shape)
    warped_norm = apply_homography(H_norm, pts_norm)
    warped_pix_from_norm = normalized_to_pixel_points(warped_norm, img2_shape)
    warped_pix = apply_homography(H_pix, pts_pix)
    assert np.allclose(warped_pix_from_norm, warped_pix, atol=1e-8)
