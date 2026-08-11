"""Audit step 1: fresh seed-123 AdelaideRMF-H run vs stored reported run."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.experiment import _evaluate_adelaide_rows_multi_policy, load_config

config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes, missing = filter_adelaide_scenes(report.scenes, "homography")
print(f"selected {len(scenes)} homography scenes, missing {missing}")

methods = list(config["methods"])
rows, _ = _evaluate_adelaide_rows_multi_policy(
    scenes, config, methods, seed=123, seed_index=0, include_outliers_values=[True]
)
fresh = pd.DataFrame(rows)
out_dir = ROOT / "outputs" / "audit_step1_seed123"
out_dir.mkdir(parents=True, exist_ok=True)
fresh.to_csv(out_dir / "fresh_seed123.csv", index=False)

print("\n=== fresh seed-123 means (include_outliers=True) ===")
print(fresh.groupby("method")[["ME_percent", "CountAcc", "AbsK"]].mean().round(3))

stored = pd.read_csv(ROOT / "outputs/adelaide_homography_20260608_214453/metrics.csv")
stored = stored[(stored.seed == 123) & (stored.include_outliers == True)]

merged = fresh.merge(
    stored[["scene_id", "method", "ME_percent", "K_est"]],
    on=["scene_id", "method"],
    suffixes=("_fresh", "_stored"),
)
merged["dME"] = (merged.ME_percent_fresh - merged.ME_percent_stored).abs()
merged["dK"] = (merged.K_est_fresh - merged.K_est_stored).abs()
print("\n=== per-scene reproduction check ===")
print(f"max |dME_percent| = {merged.dME.max():.6f}")
print(f"max |dK_est|      = {merged.dK.max():.6f}")
bad = merged[(merged.dME > 1e-9) | (merged.dK > 1e-9)]
if len(bad):
    print(bad[["scene_id", "method", "ME_percent_fresh", "ME_percent_stored", "K_est_fresh", "K_est_stored"]].to_string())
else:
    print("exact per-scene match: fresh seed-123 run reproduces the stored run")
