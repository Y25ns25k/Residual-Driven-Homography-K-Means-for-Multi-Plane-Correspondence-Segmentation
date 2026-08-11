"""Improvement A diagnostic: per-scene sigma_hat distribution on AdelaideRMF-H."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.auto_threshold import estimate_scene_sigma

report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")

rows = []
for scene in scenes:
    est = estimate_scene_sigma(scene.x1, scene.x2, random_state=123)
    rows.append(
        {
            "scene_id": scene.scene_id,
            "n_points": len(scene.x1),
            "K_gt": int(len({v for v in scene.labels if v >= 0})),
            "sigma_raw": round(est.sigma_raw, 3),
            "sigma_hat": round(est.sigma_hat, 3),
            "tau_abs_scene": round(4.0 * est.sigma_hat, 2),
            "ransac_t_scene": round(2.5 * est.sigma_hat, 2),
            "pilot_inliers": est.pilot_inliers,
        }
    )
df = pd.DataFrame(rows).sort_values("sigma_hat")
print(df.to_string(index=False))
print(f"\nsigma_hat: median={df.sigma_hat.median():.3f}, "
      f"min={df.sigma_hat.min():.3f}, max={df.sigma_hat.max():.3f}, "
      f"clipped low={int((df.sigma_raw < 0.6).sum())}, clipped high={int((df.sigma_raw > 2.5).sum())}")
