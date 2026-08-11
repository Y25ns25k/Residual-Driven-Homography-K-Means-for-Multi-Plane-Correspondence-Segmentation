"""Decouple init threshold vs assignment threshold for residual_hkm_v2.

The user's sweep couples (ransac_threshold, tau_abs, tau_norm). This completes
the 2x2: init in {base 2.5, loose 5.0} x assignment in {base 4.0/3.0, loose
7.0/4.0}, plus a zero-K-risk post-hoc variant: fit entirely at base, then only
the FINAL assignment (+ ICM) uses a looser tau.

Usage: python diagnose_decoupled_thresholds.py <variant> [seed ...]
  variants: base_init_loose_assign | loose_init_base_assign | posthoc
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
from homography_kmeans.spatial import icm_smooth_labels, knn_edges

VARIANT = sys.argv[1]
SEEDS = [int(s) for s in sys.argv[2:]] or [123, 100126, 200129, 300132, 400135]
V2 = dict(
    use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False,
    use_rank4_prior=True,
)
LAMBDA_S = 0.5
POSTHOC_TAUS = [(4.0, 3.0), (5.5, 3.5), (7.0, 4.0), (9.0, 4.5)]

config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
out_dir = ROOT / "outputs" / "decoupled_thresholds"
out_dir.mkdir(parents=True, exist_ok=True)


def make_cfg(ransac_t: float, tau_abs: float, tau_norm: float) -> dict:
    cfg = json.loads(json.dumps(config))
    cfg["ransac"]["threshold"] = ransac_t
    cfg["hkm"]["tau_abs"] = tau_abs
    cfg["hkm"]["tau_norm"] = tau_norm
    return cfg


def finalize(km, fit, x1, x2, tau_abs, tau_norm):
    """Final assignment at (tau_abs, tau_norm) + ICM, models fixed."""
    if not fit.homographies:
        return fit.labels
    errors = error_matrix(fit.homographies, x1, x2)
    scales = estimate_scales(fit.homographies, fit.labels, x1, x2, km._energy_config().sigma_min)
    labels = assign_by_residual(errors, scales, tau_abs, tau_norm, scale_adaptive=True)
    cfg_e = km._energy_config()
    from homography_kmeans.energy import EnergyConfig

    cfg_loose = EnergyConfig(
        lambda_K=cfg_e.lambda_K, gamma_outlier=cfg_e.gamma_outlier, huber_c=cfg_e.huber_c,
        sigma_min=cfg_e.sigma_min, tau_abs=tau_abs, tau_norm=tau_norm,
    )
    return icm_smooth_labels(errors, scales, labels, knn_edges(x1, k=8), cfg_loose, lambda_s=LAMBDA_S)


rows = []
for seed in SEEDS:
    t0 = time.perf_counter()
    for scene in scenes:
        x1, x2 = scene.x1, scene.x2
        image_shape = scene.image_shape or (480, 640)
        hkm_seed = _adelaide_method_seed(seed, scene.scene_id, "residual_hkm_no_merge")

        if VARIANT == "base_init_loose_assign":
            cfg = make_cfg(2.5, 7.0, 4.0)
            km = ResidualHomographyKMeans(cfg, random_state=hkm_seed, **V2)
            fit = km.fit(x1, x2, image_shape=image_shape)
            outputs = {"base_init_loose_assign": finalize(km, fit, x1, x2, 7.0, 4.0)}
        elif VARIANT == "loose_init_base_assign":
            cfg = make_cfg(5.0, 4.0, 3.0)
            km = ResidualHomographyKMeans(cfg, random_state=hkm_seed, **V2)
            fit = km.fit(x1, x2, image_shape=image_shape)
            outputs = {"loose_init_base_assign": finalize(km, fit, x1, x2, 4.0, 3.0)}
        elif VARIANT == "posthoc":
            km = ResidualHomographyKMeans(config, random_state=hkm_seed, **V2)
            fit = km.fit(x1, x2, image_shape=image_shape)
            outputs = {
                f"posthoc_tau{ta}": finalize(km, fit, x1, x2, ta, tn) for ta, tn in POSTHOC_TAUS
            }
        elif VARIANT == "posthoc_ext":
            km = ResidualHomographyKMeans(config, random_state=hkm_seed, **V2)
            fit = km.fit(x1, x2, image_shape=image_shape)
            outputs = {
                f"posthoc_tau{ta}": finalize(km, fit, x1, x2, ta, tn)
                for ta, tn in [(12.0, 5.0), (15.0, 5.5), (20.0, 6.0)]
            }
        else:
            raise SystemExit(f"unknown variant {VARIANT}")

        for name, labels in outputs.items():
            m = evaluate_segmentation(
                scene.labels, np.asarray(labels, dtype=np.int32),
                pred_homographies=fit.homographies, x1=x1, x2=x2,
                image_shape=image_shape, include_outliers=True, runtime=fit.runtime,
            )
            m["ME_percent"] = 100.0 * m["ME"]
            rows.append({"scene_id": scene.scene_id, "method": name, "seed": seed, **m})
    print(f"seed {seed} done in {time.perf_counter() - t0:.1f}s", flush=True)

tag = "_".join(str(s) for s in SEEDS)
pd.DataFrame(rows).to_csv(out_dir / f"metrics_{VARIANT}_{tag}.csv", index=False)
print(f"wrote {out_dir} ({VARIANT}, seeds {SEEDS})")
