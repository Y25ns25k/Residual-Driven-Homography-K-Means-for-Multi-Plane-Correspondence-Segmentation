from __future__ import annotations

import numpy as np

from src.homography_kmeans.metrics import evaluate_segmentation, hungarian_label_score, outlier_prf


def test_hungarian_metric_perfect_for_permuted_labels():
    gt = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    acc, me, mapping = hungarian_label_score(gt, pred)
    assert acc == 1.0
    assert me == 0.0
    assert mapping[2] == 0


def test_outlier_precision_recall_f1():
    gt = np.array([0, 0, -1, -1])
    pred = np.array([1, -1, -1, 0])
    p, r, f1 = outlier_prf(gt, pred)
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_evaluate_segmentation_k_metrics():
    gt = np.array([0, 0, 1, 1, -1])
    pred = np.array([1, 1, 0, 0, -1])
    metrics = evaluate_segmentation(gt, pred)
    assert metrics["K_gt"] == 2.0
    assert metrics["K_est"] == 2.0
    assert metrics["CountAcc"] == 1.0
    assert metrics["SegAcc"] == 1.0
