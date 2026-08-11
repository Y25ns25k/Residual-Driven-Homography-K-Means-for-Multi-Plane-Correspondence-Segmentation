from __future__ import annotations

import numpy as np

from src.geometry import apply_homography
from src.physical_consistency import (
    accept_candidate_by_physical_consistency,
    functional_distance,
    normalize_homographies,
    rank4_score,
)


def _homography(tx: float = 0.0, ty: float = 0.0, scale: float = 1.0) -> np.ndarray:
    h = np.array(
        [[scale, 0.02, tx], [-0.01, scale, ty], [0.0001, -0.0002, 1.0]],
        dtype=np.float64,
    )
    return h / h[2, 2]


def test_normalize_homographies_sign_alignment() -> None:
    h = _homography(10.0, 5.0)
    vecs = normalize_homographies([h, -h])
    assert vecs.shape == (2, 9)
    assert float(np.dot(vecs[0], vecs[1])) > 0.99


def test_rank4_score_marks_small_sets_trivial() -> None:
    hs = [_homography(float(i), float(i) * 0.5) for i in range(4)]
    score = rank4_score(hs)
    assert score.trivial
    assert score.score == 0.0


def test_rank4_score_nontrivial_for_larger_sets() -> None:
    hs = [_homography(float(i) * 7.0, float(i) * -3.0, 1.0 + 0.02 * i) for i in range(7)]
    score = rank4_score(hs)
    assert not score.trivial
    assert np.isfinite(score.score)


def test_functional_distance_zero_for_same_model() -> None:
    h = _homography(4.0, -3.0)
    grid = np.array([[0.0, 0.0], [10.0, 20.0], [100.0, 40.0]])
    assert functional_distance(h, h, grid) < 1e-9


def test_candidate_rejects_duplicate_model() -> None:
    h = _homography(10.0, 5.0)
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (30, 2))
    dst = apply_homography(h, src)
    decision = accept_candidate_by_physical_consistency(
        [h],
        h,
        src,
        dst,
        sample_points=src[:10],
        min_inliers=20,
        inlier_threshold=1.0,
        duplicate_threshold_px=0.5,
    )
    assert not decision.accepted
    assert decision.reason == "duplicate_functional_model"


def test_candidate_accepts_good_nonduplicate_when_rank4_trivial() -> None:
    existing = [_homography(0.0, 0.0), _homography(30.0, -5.0)]
    candidate = _homography(-35.0, 20.0)
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 100, (40, 2))
    dst = apply_homography(candidate, src)
    decision = accept_candidate_by_physical_consistency(
        existing,
        candidate,
        src,
        dst,
        sample_points=src[:12],
        min_inliers=20,
        inlier_threshold=1.0,
    )
    assert decision.accepted
    assert decision.reason == "accepted"
