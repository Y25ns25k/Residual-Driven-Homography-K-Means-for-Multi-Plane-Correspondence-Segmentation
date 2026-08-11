from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import apply_homography, symmetric_transfer_errors


@dataclass
class HomographyMatch:
    gt_index: int
    est_index: int
    error: float


def corner_reproj_error(h_gt: np.ndarray, h_est: np.ndarray, img_shape: Tuple[int, int]) -> float:
    height, width = img_shape[:2]
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    gt_pts = apply_homography(h_gt, corners)
    est_pts = apply_homography(h_est, corners)
    errors = np.linalg.norm(gt_pts - est_pts, axis=1)
    finite = np.isfinite(errors)
    if not np.any(finite):
        return float("inf")
    return float(np.mean(errors[finite]))


def quad_reproj_error(h_gt: np.ndarray, h_est: np.ndarray, quad: np.ndarray) -> float:
    pts = np.asarray(quad, dtype=np.float64)
    gt_pts = apply_homography(h_gt, pts)
    est_pts = apply_homography(h_est, pts)
    errors = np.linalg.norm(gt_pts - est_pts, axis=1)
    finite = np.isfinite(errors)
    if not np.any(finite):
        return float("inf")
    return float(np.mean(errors[finite]))


def auc_at_thresholds(
    errors: Iterable[float],
    thresholds: Iterable[float] = (1, 3, 5, 10),
) -> Dict[str, float]:
    arr = np.asarray(list(errors), dtype=np.float64)
    if len(arr) == 0:
        return {f"AUC@{t}": 0.0 for t in thresholds}
    return {f"AUC@{t}": float(np.mean(arr <= float(t))) for t in thresholds}


def sample_plane_labels(points: np.ndarray, plane_mask: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    mask = np.asarray(plane_mask)
    h, w = mask.shape[:2]
    xy = np.rint(pts).astype(np.int64)
    valid = (
        (xy[:, 0] >= 0) & (xy[:, 0] < w)
        & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        & np.isfinite(pts).all(axis=1)
    )
    labels = np.full(len(pts), -1, dtype=np.int32)
    labels[valid] = mask[xy[valid, 1], xy[valid, 0]].astype(np.int32)
    return labels


def plane_segmentation_accuracy(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    gt = np.asarray(gt_labels, dtype=np.int32)
    pred = np.asarray(pred_labels, dtype=np.int32)
    if gt.shape != pred.shape:
        return 0.0
    valid = gt >= 0
    if not np.any(valid):
        return 0.0
    gt = gt[valid]
    pred = pred[valid]
    gt_values = sorted(int(x) for x in np.unique(gt) if x >= 0)
    pred_values = sorted(int(x) for x in np.unique(pred) if x >= 0)
    if not gt_values or not pred_values:
        return 0.0
    confusion = np.zeros((len(gt_values), len(pred_values)), dtype=np.int64)
    gt_map = {v: i for i, v in enumerate(gt_values)}
    pred_map = {v: i for i, v in enumerate(pred_values)}
    for g, p in zip(gt, pred):
        if p < 0:
            continue
        confusion[gt_map[int(g)], pred_map[int(p)]] += 1
    row, col = linear_sum_assignment(-confusion)
    return float(confusion[row, col].sum() / len(gt))


def match_homographies(
    gt_homographies: np.ndarray,
    est_homographies: List[np.ndarray],
    img_shape: Tuple[int, int],
    source_quads: Optional[np.ndarray] = None,
) -> Tuple[List[HomographyMatch], np.ndarray]:
    gt = np.asarray(gt_homographies, dtype=np.float64)
    if len(gt) == 0 or len(est_homographies) == 0:
        return [], np.empty((len(gt), len(est_homographies)), dtype=np.float64)
    cost = np.zeros((len(gt), len(est_homographies)), dtype=np.float64)
    for i, h_gt in enumerate(gt):
        for j, h_est in enumerate(est_homographies):
            if source_quads is not None and i < len(source_quads):
                cost[i, j] = quad_reproj_error(h_gt, h_est, source_quads[i])
            else:
                cost[i, j] = corner_reproj_error(h_gt, h_est, img_shape)
    row, col = linear_sum_assignment(cost)
    matches = [HomographyMatch(int(r), int(c), float(cost[r, c])) for r, c in zip(row, col)]
    return matches, cost


def evaluate_multiplane(
    gt_homographies: np.ndarray,
    est_homographies: List[np.ndarray],
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    src_pts: Optional[np.ndarray] = None,
    dst_pts: Optional[np.ndarray] = None,
    img_shape: Tuple[int, int] = (600, 800),
    source_quads: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    matches, _ = match_homographies(
        gt_homographies, est_homographies, img_shape, source_quads=source_quads
    )
    corner_errors = [m.error for m in matches]
    valid_pred = pred_labels >= 0
    transfer_values: list[float] = []
    if src_pts is not None and dst_pts is not None:
        for k, h_est in enumerate(est_homographies):
            idx = np.flatnonzero(pred_labels == k)
            if len(idx) > 0:
                transfer_values.extend(
                    symmetric_transfer_errors(h_est, src_pts[idx], dst_pts[idx]).tolist()
                )
    return {
        "K_gt": float(len(gt_homographies)),
        "K_est": float(len(est_homographies)),
        "CountAcc": float(int(len(gt_homographies) == len(est_homographies))),
        "AbsKError": float(abs(len(gt_homographies) - len(est_homographies))),
        "SegAcc": plane_segmentation_accuracy(gt_labels, pred_labels),
        "CornerErrMean": float(np.mean(corner_errors)) if corner_errors else float("inf"),
        "CornerErrMedian": float(np.median(corner_errors)) if corner_errors else float("inf"),
        "TransferErrMean": float(np.mean(transfer_values)) if transfer_values else float("inf"),
        "TransferErrMedian": float(np.median(transfer_values)) if transfer_values else float("inf"),
        "OutlierRate": float(1.0 - np.mean(valid_pred)) if len(pred_labels) else 1.0,
        "MatchedPlanes": float(len(matches)),
    }


def misclassification_error(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    outlier_label_gt: int = 0,
    pred_outlier_label: int = -1,
) -> Dict[str, float]:
    """Compute standard AdelaideRMF Misclassification Error.

    Args:
        gt_labels: GT labels where outlier_label_gt (default 0) = gross outlier.
        pred_labels: Predicted labels where pred_outlier_label (default -1) = outlier.
        outlier_label_gt: GT label value for gross outliers.
        pred_outlier_label: Predicted label value for outliers.

    Returns dict with:
        ME_structure: misclassified / total_structure_points  (standard ME)
        ME_all: (struct_misclassified + gross_wrongly_assigned) / n_total
        SegAcc_structure: 1 - ME_structure
        structure_false_outlier_rate: structure pts predicted as outlier / structure pts
        gross_reject_rate: gross outlier pts predicted as outlier / gross pts
    """
    gt = np.asarray(gt_labels, dtype=np.int32)
    pred = np.asarray(pred_labels, dtype=np.int32)
    n = len(gt)

    structure_mask = gt != outlier_label_gt
    gross_mask = gt == outlier_label_gt
    pred_outlier_mask = pred == pred_outlier_label

    n_structure = int(np.sum(structure_mask))
    n_gross = int(np.sum(gross_mask))

    # SegAcc on structure points (Hungarian matching, ignoring pred outliers)
    seg_acc = 0.0
    if n_structure > 0:
        seg_acc = plane_segmentation_accuracy(gt[structure_mask], pred[structure_mask])

    # Structure points predicted as outlier (false negatives for structure)
    struct_false_outlier = float(np.mean(pred_outlier_mask[structure_mask])) if n_structure else 0.0

    # ME_structure: fraction of structure points not correctly assigned
    me_structure = 1.0 - seg_acc * (1.0 - struct_false_outlier)
    # More precisely: count misclassified structure points
    # = structure points that are outliers + structure points in wrong cluster
    # We compute it directly using the Hungarian matching result
    if n_structure > 0:
        gt_s = gt[structure_mask]
        pred_s = pred[structure_mask]
        # Hungarian matching
        gt_vals = sorted(int(x) for x in np.unique(gt_s) if x != outlier_label_gt)
        pred_vals = sorted(int(x) for x in np.unique(pred_s) if x != pred_outlier_label)
        if gt_vals and pred_vals:
            conf = np.zeros((len(gt_vals), len(pred_vals)), dtype=np.int64)
            gmap = {v: i for i, v in enumerate(gt_vals)}
            pmap = {v: i for i, v in enumerate(pred_vals)}
            for g, p in zip(gt_s, pred_s):
                if p != pred_outlier_label:
                    conf[gmap[int(g)], pmap[int(p)]] += 1
            row, col = linear_sum_assignment(-conf)
            correctly_matched = int(conf[row, col].sum())
        else:
            correctly_matched = 0
        # structure pts predicted as outlier contribute to misclassification
        n_pred_outlier_struct = int(np.sum(pred_outlier_mask[structure_mask]))
        me_structure = float(1.0 - (correctly_matched / n_structure))
    else:
        me_structure = 0.0

    # ME_all: includes gross outliers wrongly assigned to a structure
    gross_wrongly_assigned = int(np.sum(~pred_outlier_mask[gross_mask])) if n_gross else 0
    if n_structure > 0:
        struct_misclassified = n_structure - (correctly_matched if gt_vals and pred_vals else 0)
        me_all = float((struct_misclassified + gross_wrongly_assigned) / n)
    else:
        me_all = float(gross_wrongly_assigned / n) if n else 0.0

    gross_reject_rate = float(np.mean(pred_outlier_mask[gross_mask])) if n_gross else 1.0

    return {
        "ME_structure": me_structure,
        "ME_all": me_all,
        "SegAcc_structure": 1.0 - me_structure,
        "struct_false_outlier_rate": struct_false_outlier,
        "gross_reject_rate": gross_reject_rate,
    }
