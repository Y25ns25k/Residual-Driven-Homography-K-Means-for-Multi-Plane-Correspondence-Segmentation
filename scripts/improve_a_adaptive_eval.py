"""Improvement A evaluation: scene-adaptive thresholds vs fixed, AdelaideRMF-H.

Per scene/seed: estimate sigma_hat with a deterministic pilot fit, scale the
pixel thresholds linearly, then run sequential_ransac and
residual_hkm_no_merge with the scaled config. Seeds match the fixed-threshold
audit run exactly, so rows pair one-to-one with outputs/audit_ablation.
Usage: python improve_a_adaptive_eval.py [seed ...]
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
from homography_kmeans.auto_threshold import estimate_scene_sigma, scale_config_thresholds
from homography_kmeans.experiment import _adelaide_method_seed, load_config
from homography_kmeans.hkm import ResidualHomographyKMeans
from homography_kmeans.metrics import evaluate_segmentation
from homography_kmeans.sequential import sequential_ransac

SEEDS = [int(s) for s in sys.argv[1:]] or [123, 100126, 200129, 300132, 400135]
NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)

config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
out_dir = ROOT / "outputs" / "improve_a"
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for seed in SEEDS:
    t_seed = time.perf_counter()
    for scene in scenes:
        x1, x2 = scene.x1, scene.x2
        image_shape = scene.image_shape or (480, 640)
        seq_seed = _adelaide_method_seed(seed, scene.scene_id, "sequential_ransac")
        hkm_seed = _adelaide_method_seed(seed, scene.scene_id, "residual_hkm_no_merge")

        est = estimate_scene_sigma(x1, x2, random_state=hkm_seed + 7717)
        cfg = scale_config_thresholds(config, est.sigma_hat)
        rcfg = cfg["ransac"]

        results = {}
        t0 = time.perf_counter()
        seq = sequential_ransac(
            x1, x2,
            threshold=float(rcfg["threshold"]),
            max_iterations=int(rcfg["max_iterations"]),
            confidence=float(rcfg["confidence"]),
            min_support=int(rcfg["min_support"]),
            max_models=rcfg.get("max_models"),
            random_state=seq_seed,
        )
        results["sequential_ransac_adaptive_t"] = (list(seq.homographies), seq.labels.copy(), time.perf_counter() - t0)

        fit = ResidualHomographyKMeans(cfg, random_state=hkm_seed, **NO_MERGE).fit(x1, x2, image_shape=image_shape)
        results["residual_hkm_no_merge_adaptive_t"] = (fit.homographies, fit.labels, fit.runtime)

        for method, (Hs, labels, runtime) in results.items():
            for include_outliers in (True, False):
                metrics = evaluate_segmentation(
                    scene.labels, np.asarray(labels, dtype=np.int32),
                    pred_homographies=Hs, x1=x1, x2=x2,
                    image_shape=image_shape, include_outliers=include_outliers, runtime=runtime,
                )
                metrics["ME_percent"] = 100.0 * metrics["ME"]
                rows.append(
                    {
                        "scene_id": scene.scene_id,
                        "method": method,
                        "seed": seed,
                        "include_outliers": include_outliers,
                        "sigma_hat": est.sigma_hat,
                        "sigma_raw": est.sigma_raw,
                        **metrics,
                    }
                )
    print(f"seed {seed} done in {time.perf_counter() - t_seed:.1f}s", flush=True)

tag = "_".join(str(s) for s in SEEDS)
pd.DataFrame(rows).to_csv(out_dir / f"metrics_adaptive_seeds_{tag}.csv", index=False)
print(f"wrote {out_dir} (seeds {SEEDS})")
