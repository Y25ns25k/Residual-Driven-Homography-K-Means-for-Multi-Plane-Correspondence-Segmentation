"""Audit steps 3-4: ablation table + rank-4 denoising on AdelaideRMF-H.

Variants (all HKM variants share the residual_hkm_no_merge seed per
scene/seed so deltas isolate the component under test):

  sequential_ransac                 greedy baseline
  residual_hkm_no_merge             current best (Huber refit already on)
  residual_hkm_no_merge_adaptive    + adaptive sigma_min = 0.5*(tau_abs/3)
  residual_hkm_no_merge_binary      - Huber refit (plain DLT refit) [control]
  residual_hkm_no_merge_bic         + BIC K-selection (post-fit)
  residual_hkm_full_stack           adaptive sigma_min + Huber + BIC
  residual_hkm_no_merge_rank4       + rank-4 SVD denoising (post-fit)
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
from homography_kmeans.model_selection import select_models_bic
from homography_kmeans.rank4 import denoise_homographies_rank4
from homography_kmeans.sequential import sequential_ransac

SEEDS = [int(s) for s in sys.argv[1:]] or [123, 100126, 200129, 300132, 400135]
NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)

config = load_config(ROOT / "configs/adelaide.yml")
rcfg = config["ransac"]
hcfg = config["hkm"]
TAU_ABS = float(hcfg["tau_abs"])
TAU_NORM = float(hcfg["tau_norm"])
SIGMA_MIN = float(hcfg["sigma_min"])
ADAPTIVE_SIGMA_MIN = 0.5 * (TAU_ABS / 3.0)

report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
print(f"selected {len(scenes)} AdelaideRMF-H scenes; seeds={SEEDS}")
print(f"fixed sigma_min={SIGMA_MIN}, adaptive sigma_min={ADAPTIVE_SIGMA_MIN:.4f}")

out_dir = ROOT / "outputs" / "audit_ablation"
out_dir.mkdir(parents=True, exist_ok=True)


def residuals_for(Hs, labels, x1, x2):
    res = np.full(len(x1), np.inf, dtype=np.float64)
    if Hs:
        errors = error_matrix(Hs, x1, x2)
        valid = labels >= 0
        res[np.flatnonzero(valid)] = errors[np.flatnonzero(valid), labels[valid]]
    return res


def reassign(Hs, seed_labels, x1, x2, sigma_min):
    if not Hs:
        return np.full(len(x1), -1, dtype=np.int32)
    errors = error_matrix(Hs, x1, x2)
    scales = estimate_scales(Hs, seed_labels, x1, x2, sigma_min)
    return assign_by_residual(errors, scales, TAU_ABS, TAU_NORM, scale_adaptive=True)


rows: list[dict] = []
rank4_sv_log: list[dict] = []

for seed in SEEDS:
    t_seed = time.perf_counter()
    for scene in scenes:
        x1, x2 = scene.x1, scene.x2
        image_shape = scene.image_shape or (480, 640)
        seq_seed = _adelaide_method_seed(seed, scene.scene_id, "sequential_ransac")
        hkm_seed = _adelaide_method_seed(seed, scene.scene_id, "residual_hkm_no_merge")

        results: dict[str, tuple] = {}

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
        results["sequential_ransac"] = (list(seq.homographies), seq.labels.copy(), time.perf_counter() - t0, {})

        base = ResidualHomographyKMeans(config, random_state=hkm_seed, **NO_MERGE).fit(x1, x2, image_shape=image_shape)
        results["residual_hkm_no_merge"] = (base.homographies, base.labels, base.runtime, {})

        cfg_adaptive = json.loads(json.dumps(config))
        cfg_adaptive["hkm"]["adaptive_sigma_min"] = True
        adaptive = ResidualHomographyKMeans(cfg_adaptive, random_state=hkm_seed, **NO_MERGE).fit(x1, x2, image_shape=image_shape)
        results["residual_hkm_no_merge_adaptive"] = (adaptive.homographies, adaptive.labels, adaptive.runtime, {})

        binary = ResidualHomographyKMeans(
            config, random_state=hkm_seed, use_robust_refit=False, use_weighted_dlt=False, **NO_MERGE
        ).fit(x1, x2, image_shape=image_shape)
        results["residual_hkm_no_merge_binary"] = (binary.homographies, binary.labels, binary.runtime, {})

        t0 = time.perf_counter()
        bic_H, bic_labels, bic_info = select_models_bic(
            base.homographies, x1, x2, TAU_ABS, TAU_NORM, SIGMA_MIN, scale_adaptive=True
        )
        results["residual_hkm_no_merge_bic"] = (bic_H, bic_labels, base.runtime + time.perf_counter() - t0, bic_info)

        t0 = time.perf_counter()
        fs_H, fs_labels, fs_info = select_models_bic(
            adaptive.homographies, x1, x2, TAU_ABS, TAU_NORM, ADAPTIVE_SIGMA_MIN, scale_adaptive=True
        )
        results["residual_hkm_full_stack"] = (fs_H, fs_labels, adaptive.runtime + time.perf_counter() - t0, fs_info)

        t0 = time.perf_counter()
        den_H, den_info = denoise_homographies_rank4(base.homographies)
        r4_labels = reassign(den_H, base.labels, x1, x2, SIGMA_MIN)
        results["residual_hkm_no_merge_rank4"] = (den_H, r4_labels, base.runtime + time.perf_counter() - t0, den_info)
        rank4_sv_log.append(
            {
                "scene_id": scene.scene_id,
                "seed": seed,
                "K": den_info["K"],
                "applied": den_info["applied"],
                "singular_values": den_info["singular_values"],
            }
        )

        for method, (Hs, labels, runtime, info) in results.items():
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
                        **metrics,
                        "extra_json": json.dumps(info, default=float),
                    }
                )
    print(f"seed {seed} done in {time.perf_counter() - t_seed:.1f}s", flush=True)

tag = "_".join(str(s) for s in SEEDS)
df = pd.DataFrame(rows)
df.to_csv(out_dir / f"metrics_seeds_{tag}.csv", index=False)
with (out_dir / f"rank4_singular_values_{tag}.json").open("w", encoding="utf-8") as f:
    json.dump(rank4_sv_log, f, indent=2)
print(f"\nwrote {out_dir} (seeds {SEEDS})")
