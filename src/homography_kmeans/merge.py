from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dlt import HomographyEstimationError, estimate_homography_dlt
from .energy import EnergyConfig, compute_energy
from .geometry import functional_warp_distance


@dataclass(frozen=True)
class MergeDecision:
    accepted: bool
    i: int
    j: int
    distance: float
    delta_energy: float


def functional_merge_once(
    homographies: list[np.ndarray],
    labels: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    image_shape: tuple[int, int] | None,
    threshold: float,
    min_support: int,
    energy_config: EnergyConfig,
    energy_tolerance: float = 0.0,
    scale_adaptive: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, MergeDecision | None]:
    if len(homographies) < 2:
        return homographies, labels, None
    old_energy, _, _, _ = compute_energy(homographies, x1, x2, config=energy_config, scale_adaptive=scale_adaptive)
    best: tuple[float, float, int, int, np.ndarray, list[np.ndarray], np.ndarray] | None = None

    for i in range(len(homographies)):
        for j in range(i + 1, len(homographies)):
            if image_shape is None:
                pts = x1
                distance = functional_warp_distance(homographies[i], homographies[j], sample_points=pts)
            else:
                distance = functional_warp_distance(homographies[i], homographies[j], image_shape=image_shape)
            if distance >= threshold:
                continue
            idx = np.flatnonzero((labels == i) | (labels == j))
            if len(idx) < max(4, min_support):
                continue
            try:
                merged_H = estimate_homography_dlt(x1[idx], x2[idx])
            except HomographyEstimationError:
                continue
            candidate = [H for k, H in enumerate(homographies) if k not in {i, j}]
            insert_at = min(i, j)
            candidate.insert(insert_at, merged_H)
            new_energy, new_labels, _, _ = compute_energy(candidate, x1, x2, config=energy_config, scale_adaptive=scale_adaptive)
            delta = float(new_energy - old_energy)
            if delta <= energy_tolerance:
                if best is None or delta < best[0]:
                    best = (delta, distance, i, j, merged_H, candidate, new_labels)

    if best is None:
        return homographies, labels, None
    delta, distance, i, j, _, candidate, new_labels = best
    return candidate, new_labels.astype(np.int32), MergeDecision(True, i, j, float(distance), float(delta))


def merge_until_stable(
    homographies: list[np.ndarray],
    labels: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    image_shape: tuple[int, int] | None,
    threshold: float,
    min_support: int,
    energy_config: EnergyConfig,
    energy_tolerance: float = 0.0,
    scale_adaptive: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, list[MergeDecision]]:
    decisions: list[MergeDecision] = []
    while True:
        homographies, labels, decision = functional_merge_once(
            homographies,
            labels,
            x1,
            x2,
            image_shape,
            threshold,
            min_support,
            energy_config,
            energy_tolerance,
            scale_adaptive,
        )
        if decision is None:
            break
        decisions.append(decision)
    return homographies, labels, decisions
