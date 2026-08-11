"""Isolated min_support sweep: does lowering only the point-acceptance bar
recover barrsmith, and at what global cost (over-segmentation)?

Keeps distance threshold fixed at config default (2.5px); varies only
ransac.min_support and hkm.min_support. residual_hkm_v2, include_outliers=True.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.experiment import _adelaide_method_seed, _method_fit, load_config
from homography_kmeans.metrics import evaluate_segmentation

SEEDS = [int(s) for s in sys.argv[1:]] or [123]
MIN_SUPPORTS = [10, 12, 15, 20]

base_config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, _ = filter_adelaide_scenes(report.scenes, "homography")

rows = []
for ms in MIN_SUPPORTS:
    cfg = json.loads(json.dumps(base_config))
    cfg["ransac"]["min_support"] = ms
    cfg["hkm"]["min_support"] = ms
    for seed in SEEDS:
        for scene in scenes:
            mseed = _adelaide_method_seed(seed, scene.scene_id, "residual_hkm_no_merge")
            Hs, labels, _, _, _ = _method_fit(scene, "residual_hkm_v2", cfg, mseed)
            m = evaluate_segmentation(
                scene.labels, np.asarray(labels, dtype=np.int32),
                pred_homographies=Hs, x1=scene.x1, x2=scene.x2,
                include_outliers=True,
            )
            rows.append({
                "min_support": ms, "seed": seed, "scene_id": scene.scene_id,
                "K_gt": m["K_gt"], "K_est": m["K_est"], "ME_percent": 100 * m["ME"],
                "CountAcc": m["CountAcc"], "OverSeg": m["OverSeg"], "UnderSeg": m["UnderSeg"],
            })

df = pd.DataFrame(rows)
out = ROOT / "outputs" / "probe_min_support"
out.mkdir(parents=True, exist_ok=True)
tag = "_".join(str(s) for s in SEEDS)
df.to_csv(out / f"metrics_{tag}.csv", index=False)

print(f"seeds={SEEDS}, scenes={df.scene_id.nunique()}")
print("\n=== overall (mean over scenes x seeds) ===")
print(df.groupby("min_support")[["ME_percent", "CountAcc", "AbsK" if "AbsK" in df else "OverSeg", "OverSeg", "UnderSeg"]].mean().round(3).to_string())
print("\n=== barrsmith K_est (does the missing plane get found?) ===")
b = df[df.scene_id == "barrsmith"].groupby("min_support")[["K_gt", "K_est", "ME_percent"]].mean()
print(b.round(2).to_string())
print("\n=== over-segmentation count (K_est > K_gt), scene-seed cells per min_support ===")
print(df.assign(over=(df.K_est > df.K_gt)).groupby("min_support").over.sum().to_string())
