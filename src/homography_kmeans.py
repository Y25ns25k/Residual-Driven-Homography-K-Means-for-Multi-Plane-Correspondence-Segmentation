from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import (
    functional_homography_distance,
    grid_points_for_shape,
    symmetric_transfer_errors,
)
from .normalized_dlt import HomographyEstimationError, normalized_dlt
from .physical_consistency import accept_candidate_by_physical_consistency, rank4_score
from .ransac import estimate_homography_ransac, sequential_ransac


@dataclass
class KMeansResult:
    homographies: List[np.ndarray]
    labels: np.ndarray
    history: List[Dict[str, float]] = field(default_factory=list)
    converged: bool = False
    n_iterations: int = 0
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def n_planes(self) -> int:
        return len(self.homographies)


class HomographyKMeans:
    """Residual-driven multi-plane homography K-Means.

    This is the stable CVproject8 pipeline cleaned for the final project:
    sequential RANSAC initialization, hard assignment, robust DLT updates,
    residual discovery, and functional-distance merging. Physical consistency
    is used only to validate newly discovered plane candidates; no rank-4
    projection or denoising is applied to the final homographies.
    """

    def __init__(
        self,
        config=None,
        *,
        use_residual_discovery: bool = True,
        use_robust_refit: bool = True,
        use_relaxed_init: bool = True,
        use_weighted_dlt: bool = True,
        use_physical_validation: Optional[bool] = None,
        use_energy_discovery_accept: Optional[bool] = None,
        use_energy_merge: Optional[bool] = None,
        use_expanded_discovery_pool: Optional[bool] = None,
        max_iterations: Optional[int] = None,
        random_state: int = 42,
        # Deprecated CVproject8 rank-4 knobs are accepted for compatibility
        # but intentionally ignored in CVproject9.
        use_rank4_denoise: Optional[bool] = None,
        use_joint_refinement: Optional[bool] = None,
        use_consistency_merge: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.use_residual_discovery = use_residual_discovery
        self.use_robust_refit = use_robust_refit
        self.use_relaxed_init = use_relaxed_init
        self.use_weighted_dlt = use_weighted_dlt
        self._use_physical_validation = use_physical_validation
        self._use_energy_discovery_accept = use_energy_discovery_accept
        self._use_energy_merge = use_energy_merge
        self._use_expanded_discovery_pool = use_expanded_discovery_pool
        self.max_iterations = max_iterations
        self.random_state = random_state
        self._rank4_options_requested = any(
            value is True
            for value in (use_rank4_denoise, use_joint_refinement, use_consistency_merge)
        )

    def _cfg(self, name: str, default):
        return getattr(self.config, name, default)

    @property
    def _active_physical_validation(self) -> bool:
        if self._use_physical_validation is not None:
            return bool(self._use_physical_validation)
        return bool(self._cfg("use_physical_validation", True))

    @property
    def _active_energy_discovery_accept(self) -> bool:
        if self._use_energy_discovery_accept is not None:
            return bool(self._use_energy_discovery_accept)
        return bool(self._cfg("use_energy_discovery_accept", False))

    @property
    def _active_energy_merge(self) -> bool:
        if self._use_energy_merge is not None:
            return bool(self._use_energy_merge)
        return bool(self._cfg("use_energy_merge", False))

    @property
    def _active_expanded_discovery_pool(self) -> bool:
        if self._use_expanded_discovery_pool is not None:
            return bool(self._use_expanded_discovery_pool)
        return bool(self._cfg("use_expanded_discovery_pool", False))

    def fit(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> KMeansResult:
        src = np.asarray(src_pts, dtype=np.float64)
        dst = np.asarray(dst_pts, dtype=np.float64)
        if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
            raise ValueError("src_pts and dst_pts must have shape (N, 2)")
        n = len(src)
        if n < 4:
            return KMeansResult([], np.full(n, -1, dtype=np.int32), [], False, 0)

        diagnostics: Dict[str, object] = {
            "discovery_attempts": 0,
            "discovery_accepted": 0,
            "discovery_rejected": 0,
            "discovery_rejection_reasons": [],
            "merge_accepted": 0,
            "physical_validation_enabled": self._active_physical_validation,
            "energy_discovery_enabled": self._active_energy_discovery_accept,
            "energy_merge_enabled": self._active_energy_merge,
            "expanded_discovery_pool_enabled": self._active_expanded_discovery_pool,
            "energy_discovery_accepted": 0,
            "energy_discovery_rejected": 0,
            "energy_merge_accepted": 0,
            "energy_merge_rejected": 0,
            "energy_discovery_delta_E": [],
            "energy_merge_delta_E": [],
            "discovery_pool_outliers": 0,
            "discovery_pool_high_residual": 0,
            "discovery_pool_low_margin": 0,
            "rank4_denoise_applied": 0,
            "rank4_options_requested_ignored": self._rank4_options_requested,
        }

        homographies, labels = self._initialize(src, dst)
        if len(homographies) == 0:
            diagnostics["rank4_consistency"] = rank4_score([]).as_dict()
            return KMeansResult([], labels, [], False, 0, diagnostics)

        max_iter = int(
            self.max_iterations
            if self.max_iterations is not None
            else self._cfg("max_iterations", 50)
        )
        inlier_threshold = float(self._cfg("inlier_threshold", 2.0))
        convergence_threshold = float(self._cfg("convergence_threshold", 0.01))

        history: List[Dict[str, float]] = []
        converged = False

        for iteration in range(1, max_iter + 1):
            old_labels = labels.copy()

            errors = self._error_matrix(homographies, src, dst)
            labels = self._assign(errors, inlier_threshold)

            homographies, labels = self._update(src, dst, labels, homographies)

            if self.use_residual_discovery and len(homographies) > 0:
                homographies, labels, n_discovered = self._residual_discovery(
                    src, dst, labels, homographies, diagnostics, image_shape
                )
            else:
                n_discovered = 0

            homographies, labels, n_merged = self._merge(
                src, dst, labels, homographies, image_shape, diagnostics
            )
            diagnostics["merge_accepted"] = int(diagnostics.get("merge_accepted", 0)) + n_merged

            labels = self._compact_labels(labels)
            change = self._assignment_change(old_labels, labels)

            errors_new = self._error_matrix(homographies, src, dst) if homographies else np.empty((n, 0))
            total_error = self._total_error(errors_new, labels)
            history.append(
                {
                    "iteration": float(iteration),
                    "K": float(len(homographies)),
                    "assignment_change": change,
                    "total_error": total_error,
                    "outliers": float(np.sum(labels < 0)),
                    "discovered": float(n_discovered),
                    "merged": float(n_merged),
                }
            )

            if len(homographies) == 0 or change <= convergence_threshold:
                converged = True
                break

        diagnostics["rank4_consistency"] = rank4_score(
            homographies,
            max_rank=int(self._cfg("rank4_max_rank", 4)),
            min_planes=int(self._cfg("rank4_min_planes", 5)),
            center=bool(self._cfg("rank4_center", True)),
        ).as_dict()

        return KMeansResult(homographies, labels, history, converged, len(history), diagnostics)

    def _initialize(
        self, src: np.ndarray, dst: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        n = len(src)
        strict_threshold = float(self._cfg("init_ransac_threshold_px", 1.0))
        min_inliers = int(self._cfg("init_min_inliers", 20))
        max_planes = int(self._cfg("max_initial_planes", 4))
        max_iter = int(self._cfg("ransac_max_iter", 2500))
        confidence = float(self._cfg("ransac_confidence", 0.999))

        seq = sequential_ransac(
            src,
            dst,
            threshold=strict_threshold,
            max_iter=max_iter,
            confidence=confidence,
            min_inliers=min_inliers,
            max_planes=max_planes,
            random_state=self.random_state,
        )
        homographies = list(seq.homographies)
        labels = seq.labels.copy() if seq.homographies else np.full(n, -1, dtype=np.int32)

        use_relaxed = bool(self.use_relaxed_init and self._cfg("use_relaxed_init", True))
        if use_relaxed:
            relaxed_threshold = float(self._cfg("relaxed_init_threshold_px", 2.0))
            outlier_idx = np.flatnonzero(labels < 0)
            if len(outlier_idx) >= min_inliers:
                rng = np.random.default_rng(self.random_state + 999)
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                relaxed_seq = sequential_ransac(
                    src[outlier_idx],
                    dst[outlier_idx],
                    threshold=relaxed_threshold,
                    max_iter=max_iter,
                    confidence=confidence,
                    min_inliers=min_inliers,
                    max_planes=max(1, max_planes - len(homographies)),
                    random_state=seed,
                )
                for h, mask in zip(relaxed_seq.homographies, relaxed_seq.inlier_masks):
                    new_id = len(homographies)
                    global_inliers = outlier_idx[mask]
                    labels[global_inliers] = new_id
                    homographies.append(h)

        if not homographies:
            return [], labels

        homographies, labels = self._update(src, dst, labels, homographies)
        return homographies, self._compact_labels(labels)

    @staticmethod
    def _error_matrix(homographies: List[np.ndarray], src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        errors = np.full((len(src), len(homographies)), np.inf, dtype=np.float64)
        for k, h in enumerate(homographies):
            errors[:, k] = symmetric_transfer_errors(h, src, dst)
        return errors

    @staticmethod
    def _assign(errors: np.ndarray, inlier_threshold: float) -> np.ndarray:
        if errors.size == 0 or errors.shape[1] == 0:
            return np.full(errors.shape[0], -1, dtype=np.int32)
        labels = np.argmin(errors, axis=1).astype(np.int32)
        best = errors[np.arange(len(errors)), labels]
        labels[best > inlier_threshold] = -1
        return labels

    def _update(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        labels: np.ndarray,
        homographies: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        labels = labels.astype(np.int32, copy=True)
        new_homographies: List[np.ndarray] = []
        weight_sigma = float(self._cfg("weight_sigma", 3.0))
        refit_threshold = float(self._cfg("refit_inlier_threshold", 2.0))
        next_label = 0

        for k in range(len(homographies)):
            idx = np.flatnonzero(labels == k)
            if len(idx) < 4:
                labels[idx] = -1
                continue

            weights = None
            if self.use_weighted_dlt:
                errs = symmetric_transfer_errors(homographies[k], src[idx], dst[idx])
                weights = np.exp(-(errs**2) / (2.0 * weight_sigma**2))
            try:
                h_new = normalized_dlt(src[idx], dst[idx], weights=weights, warn_condition=False)
            except HomographyEstimationError:
                labels[idx] = -1
                continue

            if self.use_robust_refit:
                errs2 = symmetric_transfer_errors(h_new, src[idx], dst[idx])
                inlier_mask = errs2 <= refit_threshold
                if np.sum(inlier_mask) >= 4:
                    try:
                        h_new = normalized_dlt(
                            src[idx[inlier_mask]], dst[idx[inlier_mask]], warn_condition=False
                        )
                    except HomographyEstimationError:
                        pass

            new_homographies.append(h_new)
            labels[idx] = next_label
            next_label += 1

        return new_homographies, labels

    def _energy_for_homographies(
        self,
        homographies: List[np.ndarray],
        src: np.ndarray,
        dst: np.ndarray,
        inlier_threshold: Optional[float] = None,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Energy used only for exploratory model selection.

        The data term is a clipped robust symmetric-transfer loss. The model
        complexity and outlier terms discourage accepting tiny duplicate
        models that merely shave a few residuals.
        """
        threshold = float(
            inlier_threshold
            if inlier_threshold is not None
            else self._cfg("inlier_threshold", 2.0)
        )
        robust_threshold = float(self._cfg("energy_robust_threshold_px", threshold))
        if not homographies:
            labels = np.full(len(src), -1, dtype=np.int32)
            best = np.full(len(src), robust_threshold, dtype=np.float64)
        else:
            errors = self._error_matrix(homographies, src, dst)
            labels = self._assign(errors, threshold)
            best = np.min(errors, axis=1)
            best = np.where(np.isfinite(best), best, robust_threshold)

        clipped = np.minimum(best, robust_threshold)
        data_error = float(np.sum(clipped**2))
        num_outliers = int(np.sum(labels < 0))
        energy = (
            data_error
            + float(self._cfg("energy_lambda_K", 35.0)) * float(len(homographies))
            + float(self._cfg("energy_gamma_outlier", 2.0)) * float(num_outliers)
        )
        return energy, labels, {
            "data_error": data_error,
            "num_outliers": float(num_outliers),
            "K": float(len(homographies)),
        }

    def _discovery_pool(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        labels: np.ndarray,
        homographies: List[np.ndarray],
        rng: np.random.Generator,
        diagnostics: Dict[str, object],
    ) -> np.ndarray:
        outlier_idx = np.flatnonzero(labels < 0)
        diagnostics["discovery_pool_outliers"] = int(diagnostics.get("discovery_pool_outliers", 0)) + len(outlier_idx)

        if not self._active_expanded_discovery_pool or len(homographies) == 0:
            return outlier_idx

        errors = self._error_matrix(homographies, src, dst)
        assigned = np.flatnonzero(labels >= 0)
        high_residual_idx = np.empty(0, dtype=np.int64)
        if len(assigned):
            assigned_residual = errors[assigned, labels[assigned]]
            high_threshold = min(
                float(self._cfg("inlier_threshold", 2.0)),
                float(self._cfg("discovery_high_residual_factor", 1.25))
                * float(self._cfg("discovery_ransac_threshold", 1.5)),
            )
            high_residual_idx = assigned[assigned_residual > high_threshold]

        low_margin_idx = np.empty(0, dtype=np.int64)
        if errors.shape[1] >= 2:
            order = np.partition(errors, kth=1, axis=1)[:, :2]
            best = order[:, 0]
            second = order[:, 1]
            margin = second - best
            low_margin_idx = np.flatnonzero(
                (labels >= 0)
                & (best <= float(self._cfg("inlier_threshold", 2.0)))
                & (margin <= float(self._cfg("discovery_low_margin_px", 1.0)))
            )

        diagnostics["discovery_pool_high_residual"] = int(
            diagnostics.get("discovery_pool_high_residual", 0)
        ) + len(high_residual_idx)
        diagnostics["discovery_pool_low_margin"] = int(
            diagnostics.get("discovery_pool_low_margin", 0)
        ) + len(low_margin_idx)

        pool = np.unique(np.concatenate([outlier_idx, high_residual_idx, low_margin_idx]))
        max_pool = int(self._cfg("discovery_pool_max_points", 1500))
        if max_pool > 0 and len(pool) > max_pool:
            pool = np.sort(rng.choice(pool, size=max_pool, replace=False))
        return pool.astype(np.int64, copy=False)

    def _residual_discovery(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        labels: np.ndarray,
        homographies: List[np.ndarray],
        diagnostics: Dict[str, object],
        image_shape: Optional[Tuple[int, int]],
    ) -> Tuple[List[np.ndarray], np.ndarray, int]:
        discovery_min_outliers = int(self._cfg("discovery_min_outliers", 15))
        discovery_min_inliers = int(self._cfg("discovery_min_inliers", 20))
        discovery_threshold = float(self._cfg("discovery_ransac_threshold", 1.5))
        max_iter = int(self._cfg("ransac_max_iter", 2500))
        confidence = float(self._cfg("ransac_confidence", 0.999))

        n_discovered = 0
        rng = np.random.default_rng(self.random_state + 31337 + len(homographies))
        max_new_planes = int(self._cfg("discovery_max_new_planes_per_iter", 3))

        for _ in range(max_new_planes):
            pool_idx = self._discovery_pool(src, dst, labels, homographies, rng, diagnostics)
            if len(pool_idx) < discovery_min_outliers:
                break

            diagnostics["discovery_attempts"] = int(diagnostics.get("discovery_attempts", 0)) + 1
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            result = estimate_homography_ransac(
                src[pool_idx],
                dst[pool_idx],
                threshold=discovery_threshold,
                max_iter=max_iter,
                confidence=confidence,
                min_inliers=discovery_min_inliers,
                random_state=seed,
                refine=True,
            )
            if not result.success or result.homography is None or result.n_inliers < discovery_min_inliers:
                break

            if self._active_physical_validation:
                decision = accept_candidate_by_physical_consistency(
                    existing_Hs=homographies,
                    candidate_H=result.homography,
                    src_pts=src[pool_idx],
                    dst_pts=dst[pool_idx],
                    candidate_inlier_mask=result.inlier_mask,
                    sample_points=self._validation_points(src, image_shape),
                    min_inliers=discovery_min_inliers,
                    inlier_threshold=discovery_threshold,
                    duplicate_threshold_px=float(self._cfg("merge_threshold_px", 5.0)),
                    rank4_min_planes=int(self._cfg("rank4_min_planes", 5)),
                    rank4_max_rank=int(self._cfg("rank4_max_rank", 4)),
                    rank4_center=bool(self._cfg("rank4_center", True)),
                    rank4_max_score=float(self._cfg("physical_rank4_max_score", 0.35)),
                    rank4_max_delta=float(self._cfg("physical_rank4_max_delta", 0.25)),
                    max_median_error_factor=float(
                        self._cfg("physical_candidate_max_median_error_factor", 1.25)
                    ),
                )
                if not decision.accepted:
                    diagnostics["discovery_rejected"] = int(
                        diagnostics.get("discovery_rejected", 0)
                    ) + 1
                    reasons = diagnostics.setdefault("discovery_rejection_reasons", [])
                    if isinstance(reasons, list):
                        reasons.append(decision.reason)
                    break

            if self._active_energy_discovery_accept:
                old_energy, _, _ = self._energy_for_homographies(
                    homographies,
                    src,
                    dst,
                    inlier_threshold=float(self._cfg("inlier_threshold", 2.0)),
                )
                candidate_homographies = homographies + [result.homography]
                new_energy, candidate_labels, _ = self._energy_for_homographies(
                    candidate_homographies,
                    src,
                    dst,
                    inlier_threshold=float(self._cfg("inlier_threshold", 2.0)),
                )
                delta_energy = float(new_energy - old_energy)
                deltas = diagnostics.setdefault("energy_discovery_delta_E", [])
                if isinstance(deltas, list):
                    deltas.append(delta_energy)
                if delta_energy > -float(self._cfg("energy_discovery_min_delta", 0.0)):
                    diagnostics["energy_discovery_rejected"] = int(
                        diagnostics.get("energy_discovery_rejected", 0)
                    ) + 1
                    diagnostics["discovery_rejected"] = int(
                        diagnostics.get("discovery_rejected", 0)
                    ) + 1
                    reasons = diagnostics.setdefault("discovery_rejection_reasons", [])
                    if isinstance(reasons, list):
                        reasons.append(f"energy_delta={delta_energy:.3f}")
                    break

                homographies = candidate_homographies
                labels = candidate_labels.astype(np.int32, copy=True)
                n_discovered += 1
                diagnostics["discovery_accepted"] = int(diagnostics.get("discovery_accepted", 0)) + 1
                diagnostics["energy_discovery_accepted"] = int(
                    diagnostics.get("energy_discovery_accepted", 0)
                ) + 1
                continue

            new_id = len(homographies)
            global_inliers = pool_idx[result.inlier_mask]
            labels[global_inliers] = new_id
            homographies.append(result.homography)
            n_discovered += 1
            diagnostics["discovery_accepted"] = int(diagnostics.get("discovery_accepted", 0)) + 1

        return homographies, labels, n_discovered

    def _merge(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        labels: np.ndarray,
        homographies: List[np.ndarray],
        image_shape: Optional[Tuple[int, int]],
        diagnostics: Optional[Dict[str, object]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray, int]:
        if len(homographies) < 2:
            return homographies, labels, 0

        diagnostics = diagnostics if diagnostics is not None else {}
        grid = self._validation_points(src, image_shape)
        merge_threshold = float(self._cfg("merge_threshold_px", 5.0))
        parent = list(range(len(homographies)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        n_accepted = 0
        for i in range(len(homographies)):
            for j in range(i + 1, len(homographies)):
                dist = functional_homography_distance(homographies[i], homographies[j], grid)
                if dist >= merge_threshold:
                    continue
                idx = np.flatnonzero((labels == i) | (labels == j))
                if len(idx) < 4:
                    continue
                try:
                    merged_h = normalized_dlt(src[idx], dst[idx], warn_condition=False)
                except HomographyEstimationError:
                    continue
                if self._active_energy_merge:
                    old_energy, _, _ = self._energy_for_homographies(
                        homographies,
                        src,
                        dst,
                        inlier_threshold=float(self._cfg("inlier_threshold", 2.0)),
                    )
                    candidate_homographies: List[np.ndarray] = []
                    for k, h in enumerate(homographies):
                        if k == i:
                            candidate_homographies.append(merged_h)
                        elif k == j:
                            continue
                        else:
                            candidate_homographies.append(h)
                    new_energy, _, _ = self._energy_for_homographies(
                        candidate_homographies,
                        src,
                        dst,
                        inlier_threshold=float(self._cfg("inlier_threshold", 2.0)),
                    )
                    delta_energy = float(new_energy - old_energy)
                    deltas = diagnostics.setdefault("energy_merge_delta_E", [])
                    if isinstance(deltas, list):
                        deltas.append(delta_energy)
                    if delta_energy <= float(self._cfg("energy_merge_tolerance", 0.0)):
                        union(i, j)
                        n_accepted += 1
                        diagnostics["energy_merge_accepted"] = int(
                            diagnostics.get("energy_merge_accepted", 0)
                        ) + 1
                    else:
                        diagnostics["energy_merge_rejected"] = int(
                            diagnostics.get("energy_merge_rejected", 0)
                        ) + 1
                    continue

                old_errs = np.minimum(
                    symmetric_transfer_errors(homographies[i], src[idx], dst[idx]),
                    symmetric_transfer_errors(homographies[j], src[idx], dst[idx]),
                )
                new_errs = symmetric_transfer_errors(merged_h, src[idx], dst[idx])
                if np.mean(new_errs) <= 1.15 * np.mean(old_errs):
                    union(i, j)
                    n_accepted += 1

        groups: Dict[int, List[int]] = {}
        for k in range(len(homographies)):
            groups.setdefault(find(k), []).append(k)
        if len(groups) == len(homographies):
            return homographies, labels, 0

        new_labels = np.full_like(labels, -1)
        new_homographies: List[np.ndarray] = []
        for new_id, members in enumerate(groups.values()):
            idx = np.flatnonzero(np.isin(labels, members))
            if len(idx) < 4:
                continue
            try:
                h_new = normalized_dlt(src[idx], dst[idx], warn_condition=False)
            except HomographyEstimationError:
                continue
            new_homographies.append(h_new)
            new_labels[idx] = new_id
        return new_homographies, new_labels, n_accepted

    @staticmethod
    def _validation_points(
        src: np.ndarray,
        image_shape: Optional[Tuple[int, int]],
        nx: int = 6,
        ny: int = 5,
    ) -> np.ndarray:
        if image_shape is not None:
            return grid_points_for_shape(image_shape, nx=nx, ny=ny)
        min_xy = np.min(src, axis=0)
        max_xy = np.max(src, axis=0)
        xs = np.linspace(min_xy[0], max_xy[0], nx)
        ys = np.linspace(min_xy[1], max_xy[1], ny)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack([xx.ravel(), yy.ravel()])

    @staticmethod
    def _compact_labels(labels: np.ndarray) -> np.ndarray:
        compact = labels.astype(np.int32, copy=True)
        valid = sorted(int(x) for x in np.unique(compact) if x >= 0)
        for new, old in enumerate(valid):
            compact[compact == old] = new
        return compact

    @staticmethod
    def _assignment_change(old_labels: np.ndarray, labels: np.ndarray) -> float:
        if old_labels.shape != labels.shape:
            return 1.0
        if len(labels) == 0:
            return 0.0
        return float(np.mean(old_labels != labels))

    @staticmethod
    def _total_error(errors: np.ndarray, labels: np.ndarray) -> float:
        if errors.size == 0:
            return float("inf")
        valid = labels >= 0
        if not np.any(valid):
            return float("inf")
        return float(np.sum(errors[np.flatnonzero(valid), labels[valid]] ** 2))
