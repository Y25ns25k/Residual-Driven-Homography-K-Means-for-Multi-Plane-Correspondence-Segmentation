from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dlt import HomographyEstimationError, estimate_homography_dlt
from .energy import EnergyConfig, assign_by_residual, compute_energy, error_matrix, estimate_scales
from .geometry import symmetric_transfer_error
from .merge import merge_until_stable
from .rank4 import rank4_candidate_consistent
from .residual_discovery import discover_from_outliers
from .sequential import sequential_ransac
from .split import split_worst_cluster


@dataclass
class FitResult:
    homographies: list[np.ndarray]
    labels: np.ndarray
    residuals: np.ndarray
    scales: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime: float = 0.0
    converged: bool = False
    n_iterations: int = 0

    @property
    def n_planes(self) -> int:
        return len(self.homographies)


def _cfg(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _compact(labels: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    valid = [int(v) for v in sorted(np.unique(out)) if int(v) >= 0]
    for new, old in enumerate(valid):
        out[out == old] = new
    return out


def _huber_weights(u: np.ndarray, c: float) -> np.ndarray:
    a = np.abs(np.asarray(u, dtype=np.float64))
    w = np.ones_like(a)
    high = a > c
    w[high] = c / np.maximum(a[high], 1e-12)
    return np.clip(w, 1e-6, 1.0)


def _assign_conservative(
    errors: np.ndarray,
    scales: np.ndarray,
    previous_labels: np.ndarray,
    tau_abs: float,
    tau_norm: float,
    reassignment_margin: float,
    scale_adaptive: bool,
) -> np.ndarray:
    if errors.size == 0 or errors.shape[1] == 0:
        return np.full(errors.shape[0], -1, dtype=np.int32)

    normal_labels = assign_by_residual(errors, scales, tau_abs, tau_norm, scale_adaptive=scale_adaptive)
    labels = normal_labels.copy()
    prev = np.asarray(previous_labels, dtype=np.int32)
    if prev.shape != labels.shape:
        return labels

    if scale_adaptive:
        scores = errors / np.maximum(np.asarray(scales, dtype=np.float64), 1e-12)[None, :]
        best_by_score = np.argmin(scores, axis=1).astype(np.int32)
    else:
        best_by_score = np.argmin(errors, axis=1).astype(np.int32)

    for i, old_label in enumerate(prev):
        if old_label < 0 or old_label >= errors.shape[1]:
            continue
        old_error = float(errors[i, old_label])
        if scale_adaptive:
            old_score = float(scores[i, old_label])
        else:
            old_score = 0.0
        old_ok = np.isfinite(old_error) and old_error <= tau_abs and old_score <= tau_norm
        if not old_ok:
            continue
        best_label = int(best_by_score[i])
        best_error = float(errors[i, best_label])
        sufficiently_better = best_error < (1.0 - reassignment_margin) * old_error
        if normal_labels[i] >= 0 and best_label != old_label and sufficiently_better:
            labels[i] = normal_labels[i]
        else:
            labels[i] = old_label
    return labels


class ResidualHomographyKMeans:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        random_state: int = 42,
        max_iterations: int | None = None,
        use_residual_discovery: bool = True,
        use_robust_refit: bool = True,
        use_weighted_dlt: bool = True,
        use_scale_adaptive: bool = True,
        use_functional_merge: bool = True,
        use_energy_merge: bool = True,
        use_conservative_assignment: bool = False,
        use_conservative_discovery: bool = False,
        use_residual_split: bool = False,
        use_guided_discovery: bool = False,
        use_rank4_prior: bool = False,
        **legacy_ignored,
    ) -> None:
        self.config = config or {}
        self.random_state = int(random_state)
        self.max_iterations = max_iterations
        self.use_residual_discovery = bool(use_residual_discovery)
        self.use_robust_refit = bool(use_robust_refit)
        self.use_weighted_dlt = bool(use_weighted_dlt)
        self.use_scale_adaptive = bool(use_scale_adaptive)
        self.use_functional_merge = bool(use_functional_merge)
        self.use_energy_merge = bool(use_energy_merge)
        self.use_conservative_assignment = bool(use_conservative_assignment)
        self.use_conservative_discovery = bool(use_conservative_discovery)
        self.use_residual_split = bool(use_residual_split)
        self.use_guided_discovery = bool(use_guided_discovery)
        self.use_rank4_prior = bool(use_rank4_prior)
        self.legacy_ignored = legacy_ignored

    def _hkm_cfg(self) -> dict[str, Any]:
        if isinstance(self.config, dict):
            return dict(self.config.get("hkm", self.config))
        cfg = self.config
        return {
            "max_iterations": getattr(cfg, "max_iterations", 20),
            "min_support": getattr(cfg, "min_inliers", getattr(cfg, "init_min_inliers", 20)),
            "tau_abs": getattr(cfg, "inlier_threshold", 4.5),
            "tau_norm": getattr(cfg, "tau_norm", 3.0),
            "sigma_min": getattr(cfg, "sigma_min", 0.75),
            "huber_c": getattr(cfg, "huber_c", 2.5),
            "lambda_K": getattr(cfg, "energy_lambda_K", 20.0),
            "gamma_outlier": getattr(cfg, "energy_gamma_outlier", 8.0),
            "eps_energy": getattr(cfg, "energy_discovery_min_delta", 0.05),
            "energy_tolerance": getattr(cfg, "energy_merge_tolerance", 0.01),
            "label_change_tolerance": getattr(cfg, "convergence_threshold", 0.01),
            "merge_threshold": getattr(cfg, "merge_threshold_px", 4.0),
            "discovery_max_models_per_iter": getattr(cfg, "discovery_max_new_planes_per_iter", 1),
            "reassignment_margin": getattr(cfg, "reassignment_margin", 0.15),
            "discovery_improvement_margin": getattr(cfg, "discovery_improvement_margin", 0.2),
            "spatial_coverage_min": getattr(cfg, "spatial_coverage_min", 0.05),
            "discovery_split_validation": getattr(cfg, "discovery_split_validation", False),
        }

    def _ransac_cfg(self) -> dict[str, Any]:
        if isinstance(self.config, dict):
            return dict(self.config.get("ransac", {}))
        cfg = self.config
        return {
            "threshold": getattr(cfg, "init_ransac_threshold_px", 3.0),
            "max_iterations": getattr(cfg, "ransac_max_iter", 1500),
            "confidence": getattr(cfg, "ransac_confidence", 0.999),
            "min_support": getattr(cfg, "init_min_inliers", getattr(cfg, "min_inliers", 20)),
            "max_models": getattr(cfg, "max_initial_planes", None),
        }

    def _energy_config(self) -> EnergyConfig:
        c = self._hkm_cfg()
        tau_abs = float(c.get("tau_abs", c.get("inlier_threshold", 4.5)))
        sigma_min = float(c.get("sigma_min", 0.75))
        if bool(c.get("adaptive_sigma_min", False)):
            # Scale-adaptive MAD floor tied to the inlier threshold instead of
            # a fixed pixel constant: sigma_min = 0.5 * (tau_abs / 3).
            sigma_min = 0.5 * (tau_abs / 3.0)
        return EnergyConfig(
            lambda_K=float(c.get("lambda_K", 20.0)),
            gamma_outlier=float(c.get("gamma_outlier", 8.0)),
            huber_c=float(c.get("huber_c", 2.5)),
            sigma_min=sigma_min,
            tau_abs=tau_abs,
            tau_norm=float(c.get("tau_norm", 3.0)),
        )

    def _initialize(self, x1: np.ndarray, x2: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        rcfg = self._ransac_cfg()
        hcfg = self._hkm_cfg()
        seq = sequential_ransac(
            x1,
            x2,
            threshold=float(rcfg.get("threshold", hcfg.get("tau_abs", 4.5))),
            max_iterations=int(rcfg.get("max_iterations", rcfg.get("max_iter", 1500))),
            confidence=float(rcfg.get("confidence", 0.999)),
            min_support=int(rcfg.get("min_support", hcfg.get("min_support", 20))),
            max_models=rcfg.get("max_models", None),
            random_state=self.random_state,
        )
        return list(seq.homographies), seq.labels.copy()

    def _assign(self, homographies: list[np.ndarray], labels: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self._energy_config()
        hcfg = self._hkm_cfg()
        if not homographies:
            return np.full(len(x1), -1, dtype=np.int32), np.empty(0), np.empty((len(x1), 0))
        errors = error_matrix(homographies, x1, x2)
        scales = estimate_scales(homographies, labels, x1, x2, cfg.sigma_min)
        if self.use_conservative_assignment:
            new_labels = _assign_conservative(
                errors,
                scales,
                labels,
                cfg.tau_abs,
                cfg.tau_norm,
                float(hcfg.get("reassignment_margin", 0.15)),
                scale_adaptive=self.use_scale_adaptive,
            )
        else:
            new_labels = assign_by_residual(errors, scales, cfg.tau_abs, cfg.tau_norm, scale_adaptive=self.use_scale_adaptive)
        return _compact(new_labels), scales, errors

    def _update(self, homographies: list[np.ndarray], labels: np.ndarray, scales: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        hcfg = self._hkm_cfg()
        min_support = int(hcfg.get("min_support", 20))
        huber_c = float(hcfg.get("huber_c", 2.5))
        new_H: list[np.ndarray] = []
        new_labels = np.full_like(labels, -1)
        next_id = 0
        for k, H in enumerate(homographies):
            idx = np.flatnonzero(labels == k)
            if len(idx) < max(4, min_support):
                continue
            weights = None
            if self.use_robust_refit and self.use_weighted_dlt:
                residuals = symmetric_transfer_error(H, x1[idx], x2[idx])
                sigma = scales[k] if k < len(scales) else float(hcfg.get("sigma_min", 0.75))
                weights = _huber_weights(residuals / max(float(sigma), 1e-12), huber_c)
            try:
                H_new = estimate_homography_dlt(x1[idx], x2[idx], weights=weights)
            except HomographyEstimationError:
                continue
            new_H.append(H_new)
            new_labels[idx] = next_id
            next_id += 1
        return new_H, new_labels

    def fit(self, x1: np.ndarray, x2: np.ndarray, image_shape: tuple[int, int] | None = None) -> FitResult:
        t0 = time.perf_counter()
        src = np.asarray(x1, dtype=np.float64)
        dst = np.asarray(x2, dtype=np.float64)
        if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
            raise ValueError("x1 and x2 must have shape (N, 2)")
        n = len(src)
        if n < 4:
            return FitResult([], np.full(n, -1, dtype=np.int32), np.full(n, np.inf), np.empty(0), runtime=0.0)

        hcfg = self._hkm_cfg()
        rcfg = self._ransac_cfg()
        max_iter = int(self.max_iterations if self.max_iterations is not None else hcfg.get("max_iterations", 20))
        min_support = int(hcfg.get("min_support", 20))
        energy_cfg = self._energy_config()
        energy_tol = float(hcfg.get("energy_tolerance", 0.01))
        eps_energy = float(hcfg.get("eps_energy", 0.05))
        label_tol = float(hcfg.get("label_change_tolerance", hcfg.get("convergence_threshold", 0.01)))
        diagnostics: dict[str, Any] = {
            "discovery_attempts": 0,
            "discovery_accepted": 0,
            "discovery_rejected": 0,
            "merge_accepted": 0,
            "split_attempts": 0,
            "split_accepted": 0,
            "scale_adaptive": self.use_scale_adaptive,
            "conservative_assignment": self.use_conservative_assignment,
            "conservative_discovery": self.use_conservative_discovery,
            "physical_validation_enabled": False,
            "rank4_denoise_applied": 0,
            "discovery_rejection_reasons": [],
            "discovery_median_improvements": [],
            "discovery_spatial_coverages": [],
        }

        homographies, labels = self._initialize(src, dst)
        if not homographies:
            residuals = np.full(n, np.inf)
            return FitResult([], labels, residuals, np.empty(0), [], diagnostics, time.perf_counter() - t0, False, 0)

        labels, scales, errors = self._assign(homographies, labels, src, dst)
        history: list[dict[str, float]] = []
        converged = False
        prev_energy = float("inf")

        rng = np.random.default_rng(self.random_state + 917)
        for iteration in range(1, max_iter + 1):
            old_labels = labels.copy()
            homographies, labels = self._update(homographies, labels, scales, src, dst)
            labels = _compact(labels)
            labels, scales, errors = self._assign(homographies, labels, src, dst)

            if self.use_residual_discovery and homographies:
                max_new = int(hcfg.get("discovery_max_models_per_iter", 1))
                # Guided proposals vary with the sampling seed, so allow a few
                # retries per iteration before concluding nothing is found.
                attempts = max_new * (3 if self.use_guided_discovery else 1)
                accepted_this_iter = 0
                for _ in range(attempts):
                    if accepted_this_iter >= max_new:
                        break
                    diagnostics["discovery_attempts"] += 1
                    seed = int(rng.integers(0, np.iinfo(np.int32).max))
                    labels_before_discovery = labels.copy()
                    homographies2, labels2, decision = discover_from_outliers(
                        homographies,
                        labels,
                        src,
                        dst,
                        threshold=float(rcfg.get("threshold", energy_cfg.tau_abs)),
                        max_iterations=int(rcfg.get("max_iterations", 1500)),
                        confidence=float(rcfg.get("confidence", 0.999)),
                        min_support=min_support,
                        random_state=seed,
                        energy_config=energy_cfg,
                        eps_energy=eps_energy,
                        scale_adaptive=self.use_scale_adaptive,
                        image_shape=image_shape,
                        conservative=self.use_conservative_discovery or self.use_guided_discovery,
                        discovery_improvement_margin=(
                            float(hcfg.get("guided_improvement_margin", 0.1))
                            if self.use_guided_discovery
                            else float(hcfg.get("discovery_improvement_margin", 0.2))
                        ),
                        spatial_coverage_min=float(hcfg.get("spatial_coverage_min", 0.05)),
                        split_validation=bool(hcfg.get("discovery_split_validation", False))
                        and not self.use_guided_discovery,
                        local_sampling=self.use_guided_discovery,
                        local_k=int(hcfg.get("discovery_local_k", 50)),
                        from_all_points=self.use_guided_discovery,
                    )
                    diagnostics["discovery_median_improvements"].append(float(decision.median_improvement))
                    diagnostics["discovery_spatial_coverages"].append(float(decision.spatial_coverage))
                    if decision.accepted and self.use_rank4_prior and len(homographies) >= 1:
                        consistent, rank4_metrics = rank4_candidate_consistent(homographies, homographies2[-1])
                        diagnostics.setdefault("rank4_prior_checks", []).append(rank4_metrics)
                        if not consistent:
                            diagnostics["discovery_rejected"] += 1
                            diagnostics["discovery_rejection_reasons"].append("rank4_prior")
                            if not self.use_guided_discovery:
                                break
                            continue
                    if decision.accepted:
                        homographies, labels = homographies2, labels2
                        diagnostics["discovery_accepted"] += 1
                        accepted_this_iter += 1
                        if self.use_conservative_assignment:
                            labels = labels_before_discovery
                        labels, scales, errors = self._assign(homographies, labels, src, dst)
                    else:
                        diagnostics["discovery_rejected"] += 1
                        diagnostics["discovery_rejection_reasons"].append(decision.reason)
                        if not self.use_guided_discovery:
                            break

            if self.use_residual_split and homographies:
                diagnostics["split_attempts"] += 1
                split_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                homographies2, labels2, split_decision = split_worst_cluster(
                    homographies,
                    labels,
                    src,
                    dst,
                    threshold=float(rcfg.get("threshold", energy_cfg.tau_abs)),
                    max_iterations=int(rcfg.get("max_iterations", 1500)),
                    confidence=float(rcfg.get("confidence", 0.999)),
                    min_support=min_support,
                    random_state=split_seed,
                    energy_config=energy_cfg,
                    eps_energy=eps_energy,
                    scale_adaptive=self.use_scale_adaptive,
                )
                if split_decision.accepted:
                    homographies, labels = homographies2, labels2
                    diagnostics["split_accepted"] += 1
                    labels, scales, errors = self._assign(homographies, labels, src, dst)

            if self.use_functional_merge and homographies:
                homographies, labels, merge_decisions = merge_until_stable(
                    homographies,
                    labels,
                    src,
                    dst,
                    image_shape,
                    threshold=float(hcfg.get("merge_threshold", 4.0)),
                    min_support=min_support,
                    energy_config=energy_cfg,
                    energy_tolerance=0.0 if self.use_energy_merge else float("inf"),
                    scale_adaptive=self.use_scale_adaptive,
                )
                if merge_decisions:
                    diagnostics["merge_accepted"] += len(merge_decisions)
                    labels, scales, errors = self._assign(homographies, labels, src, dst)

            energy, labels, scales, parts = compute_energy(
                homographies,
                src,
                dst,
                labels=labels,
                scales=scales,
                config=energy_cfg,
                scale_adaptive=self.use_scale_adaptive,
            )
            label_change = float(np.mean(old_labels != labels)) if old_labels.shape == labels.shape else 1.0
            residuals = np.full(n, np.inf, dtype=np.float64)
            valid = labels >= 0
            if np.any(valid) and len(homographies):
                errors = error_matrix(homographies, src, dst)
                residuals[np.flatnonzero(valid)] = errors[np.flatnonzero(valid), labels[valid]]
            history.append(
                {
                    "iteration": float(iteration),
                    "K": float(len(homographies)),
                    "energy": float(energy),
                    "total_error": float(parts["data"]),
                    "outliers": float(np.sum(labels < 0)),
                    "assignment_change": label_change,
                }
            )
            if label_change <= label_tol or (prev_energy < float("inf") and prev_energy - energy <= energy_tol):
                converged = True
                break
            prev_energy = energy

        final_residuals = np.full(n, np.inf, dtype=np.float64)
        if homographies:
            final_errors = error_matrix(homographies, src, dst)
            final_labels = labels.copy()
            valid = final_labels >= 0
            final_residuals[np.flatnonzero(valid)] = final_errors[np.flatnonzero(valid), final_labels[valid]]
        else:
            final_labels = np.full(n, -1, dtype=np.int32)
            scales = np.empty(0, dtype=np.float64)
        return FitResult(
            homographies,
            final_labels,
            final_residuals,
            scales,
            history,
            diagnostics,
            time.perf_counter() - t0,
            converged,
            len(history),
        )
