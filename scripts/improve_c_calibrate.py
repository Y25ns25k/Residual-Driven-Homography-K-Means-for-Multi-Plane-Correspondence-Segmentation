"""Improvement C calibration: choose lambda_s on the synthetic suite.

Runs residual_hkm_no_merge on the synthetic full suite, then applies ICM
Potts smoothing with a grid of lambda_s values and reports mean ME
(include_outliers=True). The winning lambda_s is then frozen for Adelaide.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.energy import error_matrix, estimate_scales
from homography_kmeans.experiment import load_config
from homography_kmeans.hkm import ResidualHomographyKMeans
from homography_kmeans.metrics import evaluate_segmentation
from homography_kmeans.spatial import icm_smooth_labels, knn_edges
from homography_kmeans.synthetic import generate_synthetic_suite

NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

config = load_config(ROOT / "configs/synthetic_full.yml")
rows = []
for seed in (123, 100126, 200129):
    scenes = generate_synthetic_suite(config, seed=seed)
    for si, scene in enumerate(scenes):
        x1, x2 = scene.x1, scene.x2
        image_shape = getattr(scene, "image_shape", None) or tuple(config.get("image_shape", [480, 640]))
        km = ResidualHomographyKMeans(config, random_state=seed + 1009 * si, **NO_MERGE)
        fit = km.fit(x1, x2, image_shape=image_shape)
        if not fit.homographies:
            continue
        errors = error_matrix(fit.homographies, x1, x2)
        scales = estimate_scales(fit.homographies, fit.labels, x1, x2, km._energy_config().sigma_min)
        neighbors = knn_edges(x1, k=8)
        for lam in LAMBDAS:
            if lam == 0.0:
                labels = fit.labels
            else:
                labels = icm_smooth_labels(errors, scales, fit.labels, neighbors, km._energy_config(), lambda_s=lam)
            metrics = evaluate_segmentation(
                scene.gt_labels, np.asarray(labels, dtype=np.int32),
                pred_homographies=fit.homographies, x1=x1, x2=x2,
                image_shape=image_shape, include_outliers=True,
            )
            rows.append({"seed": seed, "scene_id": scene.scene_id, "lambda_s": lam,
                         "ME_percent": 100.0 * metrics["ME"], "SegAcc": metrics["SegAcc"],
                         "CountAcc": metrics["CountAcc"]})

df = pd.DataFrame(rows)
out = ROOT / "outputs" / "improve_c"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "synthetic_lambda_calibration.csv", index=False)
summary = df.groupby("lambda_s")[["ME_percent", "SegAcc", "CountAcc"]].mean().round(3)
print(summary.to_string())
best = summary.ME_percent.idxmin()
print(f"\nbest lambda_s on synthetic: {best}")
