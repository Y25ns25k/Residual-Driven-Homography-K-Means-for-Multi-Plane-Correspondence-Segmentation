"""Debug why split candidates are rejected on undersegmented scenes."""
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
from homography_kmeans.split import split_worst_cluster

NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)
config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")
by_id = {s.scene_id: s for s in scenes}

for scene_id in ("barrsmith", "elderhallb", "nese"):
    scene = by_id[scene_id]
    seed = _adelaide_method_seed(123, scene.scene_id, "residual_hkm_no_merge")
    km = ResidualHomographyKMeans(config, random_state=seed, **NO_MERGE)
    fit = km.fit(scene.x1, scene.x2, image_shape=scene.image_shape or (480, 640))
    counts = [int(np.sum(fit.labels == k)) for k in range(len(fit.homographies))]
    n_out = int(np.sum(fit.labels < 0))
    print(f"\n{scene_id}: n={len(scene.x1)}, K_est={len(fit.homographies)}, "
          f"cluster sizes={counts}, outliers={n_out}, scales={np.round(fit.scales, 3)}")
    for trial in range(3):
        H2, lab2, dec = split_worst_cluster(
            fit.homographies, fit.labels, scene.x1, scene.x2,
            threshold=2.5, max_iterations=2500, confidence=0.999,
            min_support=20, random_state=seed + trial,
            energy_config=km._energy_config(), eps_energy=0.05,
        )
        print(f"  trial {trial}: accepted={dec.accepted} cluster={dec.cluster} "
              f"support_a={dec.support_a} support_b={dec.support_b} "
              f"delta_energy={dec.delta_energy:+.2f} reason={dec.reason}")
