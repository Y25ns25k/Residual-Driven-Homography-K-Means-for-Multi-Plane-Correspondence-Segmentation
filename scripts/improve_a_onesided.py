"""Exact projection of one-sided adaptive thresholds (sigma_hat clipped to [1, 2.5]).

Scenes with sigma_hat <= 1 revert to the fixed config (identical thresholds
and seeds => identical results to the fixed-threshold audit run), scenes with
sigma_hat > 1 keep their measured adaptive results. So the one-sided variant
can be computed exactly from existing CSVs without a new run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
adaptive = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/improve_a").glob("metrics_adaptive_seeds_*.csv"))],
    ignore_index=True,
)
fixed = pd.read_csv(ROOT / "outputs/audit_ablation/metrics.csv")

pairs = [
    ("sequential_ransac", "sequential_ransac_adaptive_t"),
    ("residual_hkm_no_merge", "residual_hkm_no_merge_adaptive_t"),
]
for flag in (True,):
    for fixed_m, adapt_m in pairs:
        f = fixed[(fixed.method == fixed_m) & (fixed.include_outliers == flag)].copy()
        a = adaptive[(adaptive.method == adapt_m) & (adaptive.include_outliers == flag)].copy()
        sig = a[["scene_id", "seed", "sigma_hat"]]
        f = f.merge(sig, on=["scene_id", "seed"])
        use_adaptive = a.sigma_hat > 1.0 + 1e-9
        onesided = pd.concat([a[use_adaptive], f[~(f.sigma_hat > 1.0 + 1e-9)]], ignore_index=True)
        print(f"{fixed_m} (include_outliers={flag}):")
        print(f"  fixed     : ME {f.ME_percent.mean():6.2f} | CountAcc {f.CountAcc.mean():.3f} | AbsK {f.AbsK.mean():.3f}")
        print(f"  one-sided : ME {onesided.ME_percent.mean():6.2f} | CountAcc {onesided.CountAcc.mean():.3f} | AbsK {onesided.AbsK.mean():.3f}")
        print(f"  adaptive cells used: {int(use_adaptive.sum())}/{len(a)}")
