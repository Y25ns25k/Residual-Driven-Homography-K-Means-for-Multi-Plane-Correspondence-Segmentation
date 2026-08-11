from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import symmetric_transfer_error


@dataclass(frozen=True)
class EnergyConfig:
    lambda_K: float = 20.0
    gamma_outlier: float = 8.0
    huber_c: float = 2.5
    sigma_min: float = 0.75
    tau_abs: float = 4.5
    tau_norm: float = 3.0


def robust_scale_mad(residuals: np.ndarray, sigma_min: float = 0.75) -> float:
    r = np.asarray(residuals, dtype=np.float64)
    finite = r[np.isfinite(r)]
    if len(finite) == 0:
        return float(sigma_min)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return float(max(sigma_min, 1.4826 * mad))


def huber_rho(u: np.ndarray, c: float = 2.5) -> np.ndarray:
    a = np.abs(np.asarray(u, dtype=np.float64))
    return np.where(a <= c, 0.5 * a**2, c * (a - 0.5 * c))


def error_matrix(homographies: list[np.ndarray], x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    errors = np.full((len(x1), len(homographies)), np.inf, dtype=np.float64)
    for k, H in enumerate(homographies):
        errors[:, k] = symmetric_transfer_error(H, x1, x2)
    return errors


def estimate_scales(
    homographies: list[np.ndarray],
    labels: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    sigma_min: float = 0.75,
) -> np.ndarray:
    scales = np.full(len(homographies), float(sigma_min), dtype=np.float64)
    for k, H in enumerate(homographies):
        idx = np.flatnonzero(labels == k)
        if len(idx):
            scales[k] = robust_scale_mad(symmetric_transfer_error(H, x1[idx], x2[idx]), sigma_min=sigma_min)
    return scales


def assign_by_residual(
    errors: np.ndarray,
    scales: np.ndarray,
    tau_abs: float = 4.5,
    tau_norm: float = 3.0,
    scale_adaptive: bool = True,
) -> np.ndarray:
    if errors.size == 0 or errors.shape[1] == 0:
        return np.full(errors.shape[0], -1, dtype=np.int32)
    if scale_adaptive:
        safe_scales = np.maximum(np.asarray(scales, dtype=np.float64), 1e-12)
        scores = errors / safe_scales[None, :]
        best_labels = np.argmin(scores, axis=1).astype(np.int32)
        best_scores = scores[np.arange(len(errors)), best_labels]
    else:
        best_labels = np.argmin(errors, axis=1).astype(np.int32)
        best_scores = np.zeros(len(errors), dtype=np.float64)
    best_errors = errors[np.arange(len(errors)), best_labels]
    out = (best_errors > tau_abs) | (best_scores > tau_norm)
    best_labels[out | ~np.isfinite(best_errors)] = -1
    return best_labels


def compute_energy(
    homographies: list[np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
    labels: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    config: EnergyConfig | None = None,
    scale_adaptive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    cfg = config or EnergyConfig()
    if not homographies:
        lbl = np.full(len(x1), -1, dtype=np.int32)
        scl = np.empty(0, dtype=np.float64)
        return cfg.gamma_outlier * len(x1), lbl, scl, {"data": 0.0, "outliers": float(len(x1)), "K": 0.0}

    errors = error_matrix(homographies, x1, x2)
    if labels is None or scales is None:
        if scales is None:
            seed_labels = np.argmin(errors, axis=1).astype(np.int32)
            scales = estimate_scales(homographies, seed_labels, x1, x2, cfg.sigma_min)
        labels = assign_by_residual(errors, scales, cfg.tau_abs, cfg.tau_norm, scale_adaptive=scale_adaptive)
    lbl = np.asarray(labels, dtype=np.int32)
    scl = np.asarray(scales, dtype=np.float64)

    valid = lbl >= 0
    data = 0.0
    if np.any(valid):
        idx = np.flatnonzero(valid)
        residual = errors[idx, lbl[idx]]
        u = residual / np.maximum(scl[lbl[idx]], 1e-12)
        data = float(np.sum(huber_rho(u, cfg.huber_c)))
    outliers = int(np.sum(lbl < 0))
    energy = data + cfg.lambda_K * len(homographies) + cfg.gamma_outlier * outliers
    return float(energy), lbl.copy(), scl.copy(), {"data": data, "outliers": float(outliers), "K": float(len(homographies))}
