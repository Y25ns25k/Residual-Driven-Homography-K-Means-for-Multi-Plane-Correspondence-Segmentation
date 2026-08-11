"""Summarize the fairness-relaxation experiment."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/fairness_relaxed").glob("metrics_*.csv"))],
    ignore_index=True,
)
order = [
    "global_ransac",
    "global_ransac+orphan7.0",
    "sequential_ransac",
    "sequential_ransac+orphan7.0",
    "sequential_ransac+fullreassign7",
    "residual_hkm_v2",
    "residual_hkm_v2+orphan7.0",
    "residual_hkm_v2+full7_icm",
]
agg = df.groupby("method")[
    ["ME_percent", "SegAcc", "CountAcc", "AbsK", "OutlierF1"]
].mean().reindex(order)
print(f"seeds: {sorted(df.seed.unique())}, scenes: {df.scene_id.nunique()}")
print(agg.round(3).to_string())
agg.to_csv(ROOT / "outputs/fairness_relaxed/summary.csv")
