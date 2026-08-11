"""Spatially coherent label smoothing (PEARL-style Potts term, ICM solver).

Extends the existing assignment energy with a pairwise smoothness term over a
k-NN graph of the first-image coordinates:

    E(L) = sum_i unary_i(l_i) + lambda_s * sum_{(i,j) in kNN} [l_i != l_j]

where unary_i(k) is the Huber-normalized residual cost already used by
``compute_energy`` (infinite when the residual violates tau_abs/tau_norm so
threshold semantics are preserved) and unary_i(outlier) = gamma_outlier.
ICM (iterated conditional modes) is sufficient here because the initial
labels from residual assignment are already close to a good optimum.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .energy import EnergyConfig, huber_rho


def knn_edges(points: np.ndarray, k: int = 8) -> np.ndarray:
    """Return directed k-NN neighbor indices, shape (N, k)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    kk = min(int(k), n - 1)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=kk + 1)
    return np.asarray(idx[:, 1:], dtype=np.int64)


def _unary_costs(errors: np.ndarray, scales: np.ndarray, cfg: EnergyConfig) -> np.ndarray:
    """Per-point label costs, shape (N, K+1); last column is the outlier label."""
    n, K = errors.shape
    safe_scales = np.maximum(np.asarray(scales, dtype=np.float64), 1e-12)
    u = errors / safe_scales[None, :]
    costs = huber_rho(u, cfg.huber_c)
    forbidden = (errors > cfg.tau_abs) | (u > cfg.tau_norm) | ~np.isfinite(errors)
    costs = np.where(forbidden, np.inf, costs)
    out = np.full((n, K + 1), cfg.gamma_outlier, dtype=np.float64)
    out[:, :K] = costs
    return out


def icm_smooth_labels(
    errors: np.ndarray,
    scales: np.ndarray,
    labels: np.ndarray,
    neighbors: np.ndarray,
    config: EnergyConfig,
    lambda_s: float = 1.0,
    max_sweeps: int = 10,
) -> np.ndarray:
    """ICM minimization of unary + Potts energy. Outlier label is -1."""
    n, K = errors.shape
    if K == 0 or n == 0:
        return np.asarray(labels, dtype=np.int32).copy()
    unary = _unary_costs(errors, scales, config)
    # Internal label encoding: 0..K-1 models, K = outlier.
    lab = np.asarray(labels, dtype=np.int64).copy()
    lab[lab < 0] = K
    n_labels = K + 1
    for _ in range(int(max_sweeps)):
        changed = 0
        neighbor_labels = lab[neighbors] if neighbors.shape[1] else np.zeros((n, 0), dtype=np.int64)
        # Count of neighbors per candidate label, shape (N, K+1).
        agree = np.zeros((n, n_labels), dtype=np.float64)
        if neighbors.shape[1]:
            for col in range(neighbors.shape[1]):
                np.add.at(agree, (np.arange(n), neighbor_labels[:, col]), 1.0)
        pair_cost = float(lambda_s) * (neighbors.shape[1] - agree)
        total = unary + pair_cost
        new_lab = np.argmin(total, axis=1)
        changed = int(np.sum(new_lab != lab))
        lab = new_lab
        if changed == 0:
            break
    out = lab.astype(np.int32)
    out[out == K] = -1
    return out
