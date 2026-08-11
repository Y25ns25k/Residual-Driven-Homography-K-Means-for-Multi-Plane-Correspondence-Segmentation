"""BIC/MDL model-count selection over nested candidate model sets.

Candidates are built by ranking the fitted homographies by assignment support
and evaluating the top-K prefix for K = 1..K_full. Each homography carries
9 - 1 = 8 algebraic parameters, but the audit spec fixes the penalty at 9 DOF
per model, so we use ``BIC(K) = n * log(RSS(K) / n) + K * 9 * log(n)``.
Residuals of points left unassigned (outliers) are capped at ``tau_abs`` so
RSS stays finite and dropping a real plane is properly penalized.
"""
from __future__ import annotations

import numpy as np

from .energy import assign_by_residual, error_matrix, estimate_scales


def _rss(errors: np.ndarray, labels: np.ndarray, tau_abs: float) -> float:
    n = len(labels)
    r = np.full(n, float(tau_abs), dtype=np.float64)
    valid = labels >= 0
    if np.any(valid):
        idx = np.flatnonzero(valid)
        r[idx] = np.minimum(errors[idx, labels[idx]], float(tau_abs))
    return float(np.sum(r**2))


def select_models_bic(
    homographies: list[np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
    tau_abs: float = 4.0,
    tau_norm: float = 3.0,
    sigma_min: float = 0.75,
    scale_adaptive: bool = True,
    dof_per_model: int = 9,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, object]]:
    """Return the BIC-optimal model subset, its labels, and diagnostics."""
    n = len(x1)
    info: dict[str, object] = {"bic_values": [], "K_full": len(homographies), "K_selected": 0}
    if not homographies or n == 0:
        return [], np.full(n, -1, dtype=np.int32), info

    errors_full = error_matrix(homographies, x1, x2)
    scales_full = estimate_scales(
        homographies,
        np.argmin(errors_full, axis=1).astype(np.int32),
        x1,
        x2,
        sigma_min,
    )
    labels_full = assign_by_residual(errors_full, scales_full, tau_abs, tau_norm, scale_adaptive=scale_adaptive)
    support = np.array([int(np.sum(labels_full == k)) for k in range(len(homographies))])
    order = np.argsort(-support, kind="stable")

    best_bic = np.inf
    best_subset: list[np.ndarray] = []
    best_labels = np.full(n, -1, dtype=np.int32)
    log_n = float(np.log(max(n, 2)))
    for K in range(1, len(homographies) + 1):
        subset = [homographies[int(j)] for j in order[:K]]
        errors = errors_full[:, order[:K]]
        seed_labels = np.argmin(errors, axis=1).astype(np.int32)
        scales = estimate_scales(subset, seed_labels, x1, x2, sigma_min)
        labels = assign_by_residual(errors, scales, tau_abs, tau_norm, scale_adaptive=scale_adaptive)
        rss = max(_rss(errors, labels, tau_abs), 1e-12)
        bic = n * float(np.log(rss / n)) + K * int(dof_per_model) * log_n
        info["bic_values"].append({"K": K, "BIC": float(bic), "RSS": float(rss)})
        if bic < best_bic:
            best_bic = bic
            best_subset = subset
            best_labels = labels
    info["K_selected"] = len(best_subset)
    info["BIC_selected"] = float(best_bic)
    return best_subset, best_labels, info
