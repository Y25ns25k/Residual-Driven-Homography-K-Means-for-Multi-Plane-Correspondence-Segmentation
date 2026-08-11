"""Merge per-seed ablation CSVs and print the ablation summary tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "outputs" / "audit_ablation"

parts = sorted(out_dir.glob("metrics_seeds_*.csv"))
print("merging:", [p.name for p in parts])
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv(out_dir / "metrics.csv", index=False)
seeds = sorted(df.seed.unique())
print(f"rows={len(df)}, seeds={seeds}, scenes={df.scene_id.nunique()}")

sv_all = []
for p in sorted(out_dir.glob("rank4_singular_values_*.json")):
    sv_all.extend(json.loads(p.read_text(encoding="utf-8")))
(out_dir / "rank4_singular_values.json").write_text(json.dumps(sv_all, indent=2), encoding="utf-8")

order = [
    "sequential_ransac",
    "residual_hkm_no_merge",
    "residual_hkm_no_merge_adaptive",
    "residual_hkm_no_merge_binary",
    "residual_hkm_no_merge_bic",
    "residual_hkm_full_stack",
    "residual_hkm_no_merge_rank4",
]
for flag in (True, False):
    sub = df[df.include_outliers == flag]
    agg = sub.groupby("method").agg(
        ME_percent_mean=("ME_percent", "mean"),
        ME_percent_std=("ME_percent", "std"),
        SegAcc=("SegAcc", "mean"),
        CountAcc=("CountAcc", "mean"),
        AbsK=("AbsK", "mean"),
        Runtime=("Runtime", "mean"),
    ).reindex(order)
    name = "true" if flag else "false"
    agg.to_csv(out_dir / f"ablation_summary_outliers_{name}.csv")
    print(f"\n=== ablation summary (include_outliers={flag}, {len(seeds)} seeds x {sub.scene_id.nunique()} scenes) ===")
    print(agg.round(3).to_string())
