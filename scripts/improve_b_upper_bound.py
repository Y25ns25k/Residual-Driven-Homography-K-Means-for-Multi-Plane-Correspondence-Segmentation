"""Upper bound: can discovery ever succeed on the missing planes?

Fit a homography by DLT to the GT points of each missing plane (using all of
that plane's points in the scene, oracle labels) and count how many of them
fall within the discovery threshold. If that count is below min_support, no
sampling strategy can recover the plane at this threshold.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.dlt import estimate_homography_dlt
from homography_kmeans.experiment import _adelaide_method_seed, load_config
from homography_kmeans.geometry import symmetric_transfer_error
from homography_kmeans.hkm import ResidualHomographyKMeans

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
    pool = np.flatnonzero(fit.labels < 0)
    print(f"\n{scene_id}: pool={len(pool)}")
    for plane in sorted({int(v) for v in scene.labels if v >= 0}):
        gt_idx = np.flatnonzero(scene.labels == plane)
        in_pool = np.intersect1d(gt_idx, pool)
        if len(gt_idx) < 4:
            continue
        # Oracle fit on ALL of this plane's GT points (robust upper bound).
        try:
            H = estimate_homography_dlt(scene.x1[gt_idx], scene.x2[gt_idx])
        except Exception as exc:
            print(f"  plane {plane}: DLT failed ({exc})")
            continue
        r_all = symmetric_transfer_error(H, scene.x1[gt_idx], scene.x2[gt_idx])
        r_pool = symmetric_transfer_error(H, scene.x1[in_pool], scene.x2[in_pool]) if len(in_pool) else np.array([])
        print(
            f"  plane {plane}: |GT|={len(gt_idx)}, in pool={len(in_pool)}; "
            f"oracle-H inliers@2.5px: total={int(np.sum(r_all <= 2.5))}, pool-only={int(np.sum(r_pool <= 2.5)) if len(r_pool) else 0}; "
            f"@4.0px: total={int(np.sum(r_all <= 4.0))}, pool-only={int(np.sum(r_pool <= 4.0)) if len(r_pool) else 0}; "
            f"median r={np.median(r_all):.2f}px"
        )
