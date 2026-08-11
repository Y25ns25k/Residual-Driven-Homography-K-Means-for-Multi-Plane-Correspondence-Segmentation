"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    auc_at_thresholds,
    corner_reproj_error,
    evaluate_multiplane,
    match_homographies,
    plane_segmentation_accuracy,
)
from src.geometry import apply_homography


def _make_homography() -> np.ndarray:
    h = np.array([[1.1, 0.05, 20.0], [-0.03, 1.08, 15.0], [0.0002, -0.0001, 1.0]])
    return h / h[2, 2]


def test_corner_reproj_error_same_h():
    h = _make_homography()
    err = corner_reproj_error(h, h, (480, 640))
    assert err < 1e-9


def test_corner_reproj_error_identity_vs_h():
    h = _make_homography()
    err = corner_reproj_error(np.eye(3), h, (480, 640))
    assert err > 0.5


def test_auc_at_thresholds_all_below():
    errors = [0.5, 0.8, 0.9, 1.0]
    aucs = auc_at_thresholds(errors)
    assert aucs["AUC@1"] == 1.0
    assert aucs["AUC@3"] == 1.0


def test_auc_at_thresholds_mixed():
    errors = [0.5, 2.0, 6.0, 12.0]
    aucs = auc_at_thresholds(errors)
    assert aucs["AUC@1"] == 0.25
    assert aucs["AUC@3"] == 0.5
    assert aucs["AUC@5"] == 0.5
    assert aucs["AUC@10"] == 0.75


def test_seg_acc_perfect():
    gt = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    assert plane_segmentation_accuracy(gt, pred) == 1.0


def test_seg_acc_permuted():
    gt = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0])  # permuted labels → should still be 1.0
    assert plane_segmentation_accuracy(gt, pred) == 1.0


def test_seg_acc_partial():
    gt = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, -1, 1, 1, -1])  # some outliers
    acc = plane_segmentation_accuracy(gt, pred)
    assert 0 < acc < 1.0


def test_match_homographies_trivial():
    h = _make_homography()
    matches, cost = match_homographies(np.stack([h]), [h], (480, 640))
    assert len(matches) == 1
    assert matches[0].error < 1e-6


def test_evaluate_multiplane():
    h = _make_homography()
    src = np.random.default_rng(0).uniform(0, 640, (30, 2))
    dst = apply_homography(h, src)
    labels = np.zeros(30, dtype=np.int32)
    metrics = evaluate_multiplane(
        np.stack([h]), [h], labels, labels, src, dst, (480, 640)
    )
    assert metrics["CountAcc"] == 1.0
    assert metrics["SegAcc"] == 1.0
    assert metrics["K_gt"] == 1.0
    assert metrics["K_est"] == 1.0
    assert metrics["CornerErrMedian"] < 1e-6
