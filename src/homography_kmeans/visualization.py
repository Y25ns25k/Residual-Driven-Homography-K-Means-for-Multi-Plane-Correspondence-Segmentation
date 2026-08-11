from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    palette = plt.get_cmap("tab20")
    rgba = np.zeros((len(labels), 4), dtype=np.float64)
    for i, lab in enumerate(labels):
        rgba[i] = (0.25, 0.25, 0.25, 0.45) if lab < 0 else palette(int(lab) % 20)
    return rgba


def save_correspondence_plot(
    x1: np.ndarray,
    x2: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    title: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    c = _colors(labels)
    for a, b, color in zip(x1, x2, c):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=0.5, alpha=0.35)
    ax.scatter(x1[:, 0], x1[:, 1], c=c, s=10, marker="o", edgecolors="none")
    ax.scatter(x2[:, 0], x2[:, 1], c=c, s=10, marker="x")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_residual_histogram(residuals_by_method: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for name, residuals in residuals_by_method.items():
        vals = np.asarray(residuals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.45, label=name)
            plotted = True
    ax.set_xlabel("symmetric transfer error (px)")
    ax.set_ylabel("count")
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_k_error_hist(abs_k: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(abs_k, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.arange(-0.5, max(1.5, float(np.max(vals)) + 1.5), 1.0) if len(vals) else [0, 1]
    ax.hist(vals, bins=bins, color="#4c78a8")
    ax.set_xlabel("|K_est - K_gt|")
    ax.set_ylabel("scenes")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_threshold_curve(table, path: str | Path, x_col: str = "tau_abs") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric in ["ME", "SegAcc", "CountAcc", "OverSeg"]:
        if metric in table:
            grouped = table.groupby(x_col)[metric].mean()
            ax.plot(grouped.index, grouped.values, marker="o", label=metric)
    ax.set_xlabel(x_col)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
