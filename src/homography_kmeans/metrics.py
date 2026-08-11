from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import apply_homography, image_corners, symmetric_transfer_error


def _values(labels: np.ndarray) -> list[int]:
    return [int(v) for v in sorted(np.unique(labels)) if int(v) >= 0]


def hungarian_label_score(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    include_outliers: bool = False,
) -> tuple[float, float, dict[int, int]]:
    gt = np.asarray(gt_labels, dtype=np.int32)
    pred = np.asarray(pred_labels, dtype=np.int32)
    if gt.shape != pred.shape:
        raise ValueError("gt_labels and pred_labels must have same shape")
    mask = np.ones(len(gt), dtype=bool) if include_outliers else (gt >= 0)
    if not np.any(mask):
        return 0.0, 1.0, {}
    gt_eval = gt[mask]
    pred_eval = pred[mask]
    gt_vals = _values(gt_eval)
    pred_vals = _values(pred_eval)
    if include_outliers and np.any(gt_eval < 0):
        gt_vals = [-1] + gt_vals
    if include_outliers and np.any(pred_eval < 0):
        pred_vals = [-1] + pred_vals
    if not gt_vals:
        return 0.0, 1.0, {}
    if not pred_vals:
        return 0.0, 1.0, {}

    conf = np.zeros((len(gt_vals), len(pred_vals)), dtype=np.int64)
    gmap = {v: i for i, v in enumerate(gt_vals)}
    pmap = {v: i for i, v in enumerate(pred_vals)}
    for g, p in zip(gt_eval, pred_eval):
        if int(g) in gmap and int(p) in pmap:
            conf[gmap[int(g)], pmap[int(p)]] += 1
    row, col = linear_sum_assignment(-conf)
    correct = int(conf[row, col].sum())
    mapping = {pred_vals[int(c)]: gt_vals[int(r)] for r, c in zip(row, col)}
    acc = float(correct / len(gt_eval))
    return acc, 1.0 - acc, mapping


def best_label_mapping_and_correct_mask(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    include_outliers: bool = False,
) -> tuple[dict[int, int], np.ndarray]:
    """Return Hungarian pred->GT mapping and per-point correctness mask."""
    gt = np.asarray(gt_labels, dtype=np.int32)
    pred = np.asarray(pred_labels, dtype=np.int32)
    if gt.shape != pred.shape:
        raise ValueError("gt_labels and pred_labels must have same shape")
    _, _, mapping = hungarian_label_score(gt, pred, include_outliers=include_outliers)
    mapped = np.full(len(pred), -999999, dtype=np.int32)
    for label, gt_label in mapping.items():
        mapped[pred == int(label)] = int(gt_label)
    eval_mask = np.ones(len(gt), dtype=bool) if include_outliers else (gt >= 0)
    correct = np.zeros(len(gt), dtype=bool)
    correct[eval_mask] = mapped[eval_mask] == gt[eval_mask]
    return mapping, correct


def outlier_prf(gt_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[float, float, float]:
    gt_out = np.asarray(gt_labels) < 0
    pred_out = np.asarray(pred_labels) < 0
    tp = int(np.sum(gt_out & pred_out))
    fp = int(np.sum(~gt_out & pred_out))
    fn = int(np.sum(gt_out & ~pred_out))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def median_corner_transfer_error(
    gt_homographies: list[np.ndarray],
    pred_homographies: list[np.ndarray],
    image_shape: tuple[int, int],
) -> float:
    if not gt_homographies or not pred_homographies:
        return float("inf")
    corners = image_corners(image_shape)
    cost = np.zeros((len(gt_homographies), len(pred_homographies)), dtype=np.float64)
    for i, Hg in enumerate(gt_homographies):
        for j, Hp in enumerate(pred_homographies):
            eg = apply_homography(Hg, corners)
            ep = apply_homography(Hp, corners)
            d = np.linalg.norm(eg - ep, axis=1)
            cost[i, j] = np.median(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else np.inf
    row, col = linear_sum_assignment(cost)
    vals = cost[row, col]
    finite = vals[np.isfinite(vals)]
    return float(np.median(finite)) if len(finite) else float("inf")


def evaluate_segmentation(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_homographies: list[np.ndarray] | None = None,
    x1: np.ndarray | None = None,
    x2: np.ndarray | None = None,
    gt_homographies: list[np.ndarray] | None = None,
    image_shape: tuple[int, int] | None = None,
    include_outliers: bool = False,
    runtime: float | None = None,
) -> dict[str, float]:
    gt = np.asarray(gt_labels, dtype=np.int32)
    pred = np.asarray(pred_labels, dtype=np.int32)
    K_gt = len(_values(gt))
    K_est = len(_values(pred))
    seg_acc, me, _ = hungarian_label_score(gt, pred, include_outliers=include_outliers)
    op, or_, of1 = outlier_prf(gt, pred)
    median_transfer = float("inf")
    if pred_homographies is not None and x1 is not None and x2 is not None:
        vals: list[float] = []
        for k, H in enumerate(pred_homographies):
            idx = np.flatnonzero(pred == k)
            if len(idx):
                vals.extend(symmetric_transfer_error(H, x1[idx], x2[idx]).tolist())
        finite = np.asarray(vals, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            median_transfer = float(np.median(finite))
    corner = float("nan")
    if gt_homographies is not None and pred_homographies is not None and image_shape is not None:
        corner = median_corner_transfer_error(gt_homographies, pred_homographies, image_shape)
    return {
        "K_gt": float(K_gt),
        "K_est": float(K_est),
        "AbsK": float(abs(K_est - K_gt)),
        "CountAcc": float(K_est == K_gt),
        "OverSeg": float(K_est > K_gt),
        "UnderSeg": float(K_est < K_gt),
        "SegAcc": float(seg_acc),
        "ME": float(me),
        "OutlierPrecision": op,
        "OutlierRecall": or_,
        "OutlierF1": of1,
        "MedianTransferErr": median_transfer,
        "MedianCornerErr": corner,
        "Runtime": float(runtime if runtime is not None else 0.0),
    }
