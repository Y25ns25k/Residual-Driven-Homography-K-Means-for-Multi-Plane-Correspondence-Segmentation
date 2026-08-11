"""Corrected rank-4 variant: reassign only when the projection is non-trivial.

For K <= 4 the rank-4 projection is exact (no-op), so the fit's labels are
kept unchanged; only K >= 5 fits get denoised models + residual reassignment.
"""
from __future__ import annotations

import json
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
from homography_kmeans.energy import assign_by_residual, error_matrix, estimate_scales
from homography_kmeans.experiment import _adelaide_method_seed, load_config
from homography_kmeans.hkm import ResidualHomographyKMeans
from homography_kmeans.metrics import evaluate_segmentation
from homography_kmeans.rank4 import denoise_homographies_rank4

SEEDS = [123, 100126, 200129, 300132, 400135]
NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)

config = load_config(ROOT / "configs/adelaide.yml")
TAU_ABS = float(config["hkm"]["tau_abs"])
TAU_NORM = float(config["hkm"]["tau_norm"])
SIGMA_MIN = float(config["hkm"]["sigma_min"])

report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
out_dir = ROOT / "outputs" / "audit_ablation"

rows = []
for seed in SEEDS:
    t_seed = time.perf_counter()
    for scene in scenes:
        x1, x2 = scene.x1, scene.x2
        image_shape = scene.image_shape or (480, 640)
        hkm_seed = _adelaide_method_seed(seed, scene.scene_id, "residual_hkm_no_merge")
        base = ResidualHomographyKMeans(config, random_state=hkm_seed, **NO_MERGE).fit(x1, x2, image_shape=image_shape)
        t0 = time.perf_counter()
        den_H, den_info = denoise_homographies_rank4(base.homographies)
        if den_info["applied"]:
            errors = error_matrix(den_H, x1, x2)
            scales = estimate_scales(den_H, base.labels, x1, x2, SIGMA_MIN)
            labels = assign_by_residual(errors, scales, TAU_ABS, TAU_NORM, scale_adaptive=True)
        else:
            labels = base.labels
        runtime = base.runtime + time.perf_counter() - t0
        for include_outliers in (True, False):
            metrics = evaluate_segmentation(
                scene.labels, np.asarray(labels, dtype=np.int32),
                pred_homographies=den_H, x1=x1, x2=x2,
                image_shape=image_shape, include_outliers=include_outliers, runtime=runtime,
            )
            metrics["ME_percent"] = 100.0 * metrics["ME"]
            rows.append(
                {
                    "scene_id": scene.scene_id,
                    "method": "residual_hkm_no_merge_rank4",
                    "seed": seed,
                    "include_outliers": include_outliers,
                    **metrics,
                    "extra_json": json.dumps(den_info, default=float),
                }
            )
    print(f"seed {seed} done in {time.perf_counter() - t_seed:.1f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(out_dir / "metrics_rank4_corrected.csv", index=False)
for flag in (True, False):
    sub = df[df.include_outliers == flag]
    print(f"\n=== corrected rank4 variant (include_outliers={flag}) ===")
    print(
        sub.agg(
            ME_percent_mean=("ME_percent", "mean"),
            ME_percent_std=("ME_percent", "std"),
            SegAcc=("SegAcc", "mean"),
            CountAcc=("CountAcc", "mean"),
            AbsK=("AbsK", "mean"),
        ).round(3).to_string()
    )
