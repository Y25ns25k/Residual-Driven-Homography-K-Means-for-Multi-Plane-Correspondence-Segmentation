"""Audit step 1 helper: stored seed-123 and 5-seed means from the reported run."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "outputs/adelaide_homography_20260608_214453/metrics.csv")

print("=== stored run: seed 123 only, include_outliers=True ===")
sub = df[(df.seed == 123) & (df.include_outliers == True)]
print(sub.groupby("method")[["ME_percent", "CountAcc", "AbsK"]].mean().round(3))
print(f"rows: {len(sub)}, scenes: {sub.scene_id.nunique()}")

print()
print("=== stored run: all 5 seeds, include_outliers=True ===")
full = df[df.include_outliers == True]
print(full.groupby("method")[["ME_percent", "CountAcc", "AbsK"]].mean().round(3))
