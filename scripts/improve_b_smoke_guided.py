"""Smoke test: guided (local-sampling) discovery on undersegmented scenes."""
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
from homography_kmeans.metrics import evaluate_segmentation

NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)
config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
by_id = {s.scene_id: s for s in scenes}

for scene_id in ("barrsmith", "elderhallb", "nese", "bonython", "bonhall"):
    scene = by_id[scene_id]
    seed = _adelaide_method_seed(123, scene.scene_id, "residual_hkm_no_merge")
    for guided in (False, True):
        km = ResidualHomographyKMeans(config, random_state=seed, use_guided_discovery=guided, **NO_MERGE)
        fit = km.fit(scene.x1, scene.x2, image_shape=scene.image_shape or (480, 640))
        m = evaluate_segmentation(scene.labels, fit.labels, pred_homographies=fit.homographies,
                                  x1=scene.x1, x2=scene.x2, include_outliers=True, runtime=fit.runtime)
        d = fit.diagnostics
        print(f"{scene_id:>12} guided={guided}: K_gt={int(m['K_gt'])} K_est={int(m['K_est'])} "
              f"ME={100*m['ME']:.2f}% disc_attempts={d['discovery_attempts']} "
              f"accepted={d['discovery_accepted']} runtime={fit.runtime:.1f}s")
