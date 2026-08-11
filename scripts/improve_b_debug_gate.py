"""Find which conservative gate rejects the elderhallb / bonhall proposals."""
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

for scene_id in ("elderhallb", "bonhall"):
    scene = by_id[scene_id]
    seed = _adelaide_method_seed(123, scene.scene_id, "residual_hkm_no_merge")
    km = ResidualHomographyKMeans(config, random_state=seed, **NO_MERGE)
    fit = km.fit(scene.x1, scene.x2, image_shape=scene.image_shape or (480, 640))
    print(f"\n{scene_id} (n={len(scene.x1)}, K_est={len(fit.homographies)}):")
    for sv in (False, True):
        for trial in range(4):
            _, _, dec = discover_from_outliers(
                fit.homographies, fit.labels, scene.x1, scene.x2,
                threshold=2.5, max_iterations=2500, confidence=0.999,
                min_support=20, random_state=seed + 31 * trial,
                energy_config=km._energy_config(), eps_energy=0.05,
                image_shape=scene.image_shape or (480, 640),
                conservative=True, discovery_improvement_margin=0.2,
                spatial_coverage_min=0.05, split_validation=sv,
                local_sampling=True, from_all_points=True,
            )
            print(f"  split_val={sv} trial {trial}: accepted={dec.accepted} support={dec.support} "
                  f"improvement={dec.median_improvement:.3f} coverage={dec.spatial_coverage:.3f} "
                  f"dE={dec.delta_energy:+.1f} reason={dec.reason}")
