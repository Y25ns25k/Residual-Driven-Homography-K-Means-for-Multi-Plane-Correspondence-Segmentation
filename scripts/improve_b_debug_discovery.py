"""Debug why outlier-pool discovery fails on barrsmith-type scenes."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.experiment import _adelaide_method_seed, load_config
from homography_kmeans.hkm import ResidualHomographyKMeans
from homography_kmeans.residual_discovery import discover_from_outliers

NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)
config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
by_id = {s.scene_id: s for s in scenes}

for scene_id in ("barrsmith", "elderhallb"):
    scene = by_id[scene_id]
    seed = _adelaide_method_seed(123, scene.scene_id, "residual_hkm_no_merge")
    km = ResidualHomographyKMeans(config, random_state=seed, **NO_MERGE)
    fit = km.fit(scene.x1, scene.x2, image_shape=scene.image_shape or (480, 640))
    out_idx = np.flatnonzero(fit.labels < 0)
    gt_out = scene.labels[out_idx]
    uniq, cnts = np.unique(gt_out, return_counts=True)
    print(f"\n{scene_id}: outlier pool {len(out_idx)} points; GT composition: "
          + ", ".join(f"label {u}: {c}" for u, c in zip(uniq, cnts)))
    print(f"  fit history: {[(h['iteration'], h['K'], round(h['energy'],1)) for h in fit.history]}")
    for trial in range(5):
        H2, lab2, dec = discover_from_outliers(
            fit.homographies, fit.labels, scene.x1, scene.x2,
            threshold=2.5, max_iterations=2500, confidence=0.999,
            min_support=20, random_state=seed + 31 * trial,
            energy_config=km._energy_config(), eps_energy=0.05,
        )
        print(f"  trial {trial}: accepted={dec.accepted} support={dec.support} "
              f"delta_energy={dec.delta_energy:+.2f} reason={dec.reason}")
