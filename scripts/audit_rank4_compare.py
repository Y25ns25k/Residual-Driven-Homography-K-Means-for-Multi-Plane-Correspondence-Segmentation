"""Per-fit comparison: no_merge vs corrected rank-4 variant on K>=5 fits."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "outputs" / "audit_ablation"

base = pd.read_csv(out_dir / "metrics.csv")
base = base[(base.method == "residual_hkm_no_merge") & (base.include_outliers == True)]
r4 = pd.read_csv(out_dir / "metrics_rank4_corrected.csv")
r4 = r4[r4.include_outliers == True].copy()
r4["applied"] = r4.extra_json.apply(lambda s: bool(json.loads(s)["applied"]))

m = base.merge(r4, on=["scene_id", "seed"], suffixes=("_base", "_r4"))
applied = m[m.applied]
print(f"fits where rank-4 projection applied (K>=5): {len(applied)}/{len(m)}")
print(applied[["scene_id", "seed", "K_gt_base", "K_est_base", "K_est_r4",
               "ME_percent_base", "ME_percent_r4"]].round(2).to_string(index=False))
delta = applied.ME_percent_r4 - applied.ME_percent_base
print(f"\nmean delta ME on applied fits: {delta.mean():+.2f} pp "
      f"(min {delta.min():+.2f}, max {delta.max():+.2f})")
not_applied = m[~m.applied]
d2 = (not_applied.ME_percent_r4 - not_applied.ME_percent_base).abs().max()
print(f"max |delta ME| on no-op fits (should be 0): {d2:.6f}")
