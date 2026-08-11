"""Fairness check: apply the relaxed final-labeling convention to ALL methods.

Convention A (uniform, method-agnostic): each method produces (models, labels)
with its own pipeline; afterwards, points still labeled outlier are relabeled
to the nearest model iff its residual <= TAU_F. Existing labels are untouched.

Also measured: seq + FULL reassignment at the loose tau (which is literally
the first HKM assignment step) to show where the baseline/method boundary is,
and v2 + full reassignment + ICM (the earlier posthoc variant).

Usage: python fairness_relaxed_labeling.py [seed ...]
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.energy import EnergyConfig, assign_by_residual, error_matrix, estimate_scales
from homography_kmeans.experiment import _adelaide_method_seed, _method_fit, load_config
from homography_kmeans.metrics import evaluate_segmentation
from homography_kmeans.spatial import icm_smooth_labels, knn_edges

SEEDS = [int(s) for s in sys.argv[1:]] or [123, 100126, 200129, 300132, 400135]
TAU_F = 7.0
TAU_NORM_F = 4.0
SIGMA_MIN = 0.75
LAMBDA_S = 0.5

config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
out_dir = ROOT / "outputs" / "fairness_relaxed"
out_dir.mkdir(parents=True, exist_ok=True)


def orphan_relabel(Hs, labels, x1, x2, tau):
    """Convention A: relabel only outliers to the nearest model within tau."""
    out = np.asarray(labels, dtype=np.int32).copy()
    if not Hs:
        return out
    orphans = np.flatnonzero(out < 0)
    if not len(orphans):
        return out
    errors = error_matrix(Hs, x1[orphans], x2[orphans])
    best = np.argmin(errors, axis=1)
    best_err = errors[np.arange(len(orphans)), best]
    take = best_err <= tau
    out[orphans[take]] = best[take].astype(np.int32)
    return out


def full_reassign(Hs, labels, x1, x2, tau, tau_norm, with_icm=False):
    if not Hs:
        return np.asarray(labels, dtype=np.int32).copy()
    errors = error_matrix(Hs, x1, x2)
    scales = estimate_scales(Hs, labels, x1, x2, SIGMA_MIN)
    new = assign_by_residual(errors, scales, tau, tau_norm, scale_adaptive=True)
    if with_icm:
        cfg = EnergyConfig(sigma_min=SIGMA_MIN, tau_abs=tau, tau_norm=tau_norm)
        new = icm_smooth_labels(errors, scales, new, knn_edges(x1, k=8), cfg, lambda_s=LAMBDA_S)
    return new


rows = []
for seed in SEEDS:
    t_seed = time.perf_counter()
    for scene in scenes:
        x1, x2 = scene.x1, scene.x2
        image_shape = scene.image_shape or (480, 640)
        for method in ("global_ransac", "sequential_ransac", "residual_hkm_v2"):
            mseed = _adelaide_method_seed(seed, scene.scene_id, method)
            Hs, labels, _, runtime, _ = _method_fit(scene, method, config, mseed)
            labels = np.asarray(labels, dtype=np.int32)
            variants = {
                f"{method}": labels,
                f"{method}+orphan{TAU_F}": orphan_relabel(Hs, labels, x1, x2, TAU_F),
            }
            if method == "sequential_ransac":
                variants["sequential_ransac+fullreassign7"] = full_reassign(Hs, labels, x1, x2, TAU_F, TAU_NORM_F)
            if method == "residual_hkm_v2":
                variants["residual_hkm_v2+full7_icm"] = full_reassign(Hs, labels, x1, x2, TAU_F, TAU_NORM_F, with_icm=True)
            for name, lab in variants.items():
                m = evaluate_segmentation(
                    scene.labels, lab, pred_homographies=Hs, x1=x1, x2=x2,
                    image_shape=image_shape, include_outliers=True, runtime=runtime,
                )
                m["ME_percent"] = 100.0 * m["ME"]
                rows.append({"scene_id": scene.scene_id, "method": name, "seed": seed, **m})
    print(f"seed {seed} done in {time.perf_counter() - t_seed:.1f}s", flush=True)

tag = "_".join(str(s) for s in SEEDS)
pd.DataFrame(rows).to_csv(out_dir / f"metrics_{tag}.csv", index=False)
print(f"wrote {out_dir} (seeds {SEEDS})")
