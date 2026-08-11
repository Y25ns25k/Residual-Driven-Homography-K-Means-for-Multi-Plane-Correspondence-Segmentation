from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import imwrite


COLORS = np.array(
    [
        [0.90, 0.15, 0.15],
        [0.15, 0.55, 0.90],
        [0.15, 0.75, 0.35],
        [0.95, 0.65, 0.10],
        [0.60, 0.35, 0.85],
        [0.10, 0.75, 0.75],
    ],
    dtype=np.float64,
)


def label_to_color(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.full((*arr.shape, 3), 36, dtype=np.uint8)
    for label in sorted(int(x) for x in np.unique(arr) if x >= 0):
        out[arr == label] = np.round(COLORS[label % len(COLORS)] * 255).astype(np.uint8)
    return out


def save_label_overlay(image: np.ndarray, labels: np.ndarray, path: str | Path, alpha: float = 0.42) -> None:
    colors = label_to_color(labels)
    overlay = cv2.addWeighted(image, 1.0 - alpha, colors, alpha, 0.0)
    imwrite(path, overlay)


def save_match_plot(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    max_lines: int = 160,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    h = max(image1.shape[0], image2.shape[0])
    canvas = np.zeros((h, image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)
    canvas[: image1.shape[0], : image1.shape[1]] = image1
    canvas[: image2.shape[0], image1.shape[1] :] = image2
    idx = np.arange(len(pts1))
    if len(idx) > max_lines:
        idx = np.linspace(0, len(idx) - 1, max_lines).astype(np.int64)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    for i in idx:
        label = int(labels[i]) if len(labels) > i else -1
        color = COLORS[label % len(COLORS)] if label >= 0 else np.array([0.4, 0.4, 0.4])
        ax.plot(
            [pts1[i, 0], pts2[i, 0] + image1.shape[1]],
            [pts1[i, 1], pts2[i, 1]],
            color=color, lw=0.8, alpha=0.8,
        )
        ax.scatter([pts1[i, 0], pts2[i, 0] + image1.shape[1]], [pts1[i, 1], pts2[i, 1]], s=7, color=color)
    ax.axis("off")
    fig.savefig(path_obj, dpi=160)
    plt.close(fig)


def scatter_by_labels(
    ax,
    image: np.ndarray,
    pts: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for label in sorted(int(x) for x in np.unique(labels) if x >= 0):
        idx = labels == label
        color = COLORS[label % len(COLORS)]
        ax.scatter(pts[idx, 0], pts[idx, 1], s=6, color=color, alpha=0.85, linewidths=0)
    outliers = labels < 0
    if np.any(outliers):
        ax.scatter(pts[outliers, 0], pts[outliers, 1], s=5, color="0.25", alpha=0.5, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_segmentation_figure(
    source: np.ndarray,
    target: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    source_mask: np.ndarray,
    path: str | Path,
    title: str = "",
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    scatter_by_labels(axes[0, 0], source, src_pts, gt_labels, "GT labels")
    scatter_by_labels(axes[0, 1], source, src_pts, pred_labels, "Predicted labels")
    axes[1, 0].imshow(cv2.cvtColor(label_to_color(source_mask), cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("GT source mask")
    axes[1, 0].axis("off")
    # Correspondence lines
    h = max(source.shape[0], target.shape[0])
    canvas = np.zeros((h, source.shape[1] + target.shape[1], 3), dtype=np.uint8)
    canvas[: source.shape[0], : source.shape[1]] = source
    canvas[: target.shape[0], source.shape[1] :] = target
    axes[1, 1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    idx = np.arange(len(src_pts))
    if len(idx) > 180:
        idx = np.linspace(0, len(idx) - 1, 180).astype(np.int64)
    for i in idx:
        label = int(pred_labels[i])
        color = COLORS[label % len(COLORS)] if label >= 0 else np.array([0.25, 0.25, 0.25])
        axes[1, 1].plot(
            [src_pts[i, 0], dst_pts[i, 0] + source.shape[1]],
            [src_pts[i, 1], dst_pts[i, 1]],
            color=color, lw=0.55, alpha=0.75,
        )
    axes[1, 1].set_title("Predicted correspondences")
    axes[1, 1].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path_obj, dpi=170)
    plt.close(fig)


def plot_convergence(
    history: List[dict],
    path: str | Path,
    title: str = "",
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    iterations = [entry["iteration"] for entry in history]
    k_vals = [entry["K"] for entry in history]
    errors = [entry["total_error"] for entry in history]
    outliers = [entry["outliers"] for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(iterations, k_vals, "o-")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("K (planes)")
    axes[0].set_title("K over iterations")
    axes[1].plot(iterations, errors, "o-", color="tab:orange")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Total transfer error")
    axes[1].set_title("Error convergence")
    axes[2].plot(iterations, outliers, "o-", color="tab:red")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("# outliers")
    axes[2].set_title("Outliers over iterations")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path_obj, dpi=150)
    plt.close(fig)
