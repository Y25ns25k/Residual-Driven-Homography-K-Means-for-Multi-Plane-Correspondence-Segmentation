"""Improvement C evaluation: ICM Potts smoothing on AdelaideRMF-H.

lambda_s = 0.5 was frozen on the synthetic suite (see improve_c_calibrate).
0.25 and 1.0 are reported as sensitivity rows only.
Usage: python improve_c_adelaide_eval.py [seed ...]
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
from homography_kmeans.energy import error_matrix, estimate_scales
from homography_kmeans.experiment import _adelaide_method_seed, load_config
from homography_kmeans.hkm import ResidualHomographyKMeans
from homography_kmeans.metrics import evaluate_segmentation
from homography_kmeans.spatial import icm_smooth_labels, knn_edges

SEEDS = [int(s) for s in sys.argv[1:]] or [123, 100126, 200129, 300132, 400135]
NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)
LAMBDAS = [0.25, 0.5, 1.0]

config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
out_dir = ROOT / "outputs" / "improve_c"
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for seed in SEEDS:
    t_seed = time.perf_counter()
    for scene in scenes:
        x1, x2 = scene.x1, scene.x2
        image_shape = scene.image_shape or (480, 640)
        hkm_seed = _adelaide_method_seed(seed, scene.scene_id, "residual_hkm_no_merge")
        km = ResidualHomographyKMeans(config, random_state=hkm_seed, **NO_MERGE)
        fit = km.fit(x1, x2, image_shape=image_shape)
        if fit.homographies:
            errors = error_matrix(fit.homographies, x1, x2)
            scales = estimate_scales(fit.homographies, fit.labels, x1, x2, km._energy_config().sigma_min)
            neighbors = knn_edges(x1, k=8)
        for lam in LAMBDAS:
            t0 = time.perf_counter()
            if fit.homographies:
                labels = icm_smooth_labels(errors, scales, fit.labels, neighbors, km._energy_config(), lambda_s=lam)
            else:
                labels = fit.labels
            runtime = fit.runtime + time.perf_counter() - t0
            for include_outliers in (True, False):
                metrics = evaluate_segmentation(
                    scene.labels, np.asarray(labels, dtype=np.int32),
                    pred_homographies=fit.homographies, x1=x1, x2=x2,
                    image_shape=image_shape, include_outliers=include_outliers, runtime=runtime,
                )
                metrics["ME_percent"] = 100.0 * metrics["ME"]
                rows.append(
                    {
                        "scene_id": scene.scene_id,
                        "method": f"residual_hkm_no_merge_icm{lam}",
                        "seed": seed,
                        "include_outliers": include_outliers,
                        "lambda_s": lam,
                        **metrics,
                    }
                )
    print(f"seed {seed} done in {time.perf_counter() - t_seed:.1f}s", flush=True)

tag = "_".join(str(s) for s in SEEDS)
pd.DataFrame(rows).to_csv(out_dir / f"metrics_icm_seeds_{tag}.csv", index=False)
print(f"wrote {out_dir} (seeds {SEEDS})")
