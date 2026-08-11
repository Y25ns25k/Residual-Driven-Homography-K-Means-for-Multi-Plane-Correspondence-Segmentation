"""Improvement A analysis: adaptive thresholds vs fixed (paired per scene)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

ROOT = Path(__file__).resolve().parents[1]
adaptive = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/improve_a").glob("metrics_adaptive_seeds_*.csv"))],
    ignore_index=True,
)
fixed = pd.read_csv(ROOT / "outputs/audit_ablation/metrics.csv")

print(f"adaptive rows={len(adaptive)}, seeds={sorted(adaptive.seed.unique())}")

pairs = [
    ("sequential_ransac", "sequential_ransac_adaptive_t"),
    ("residual_hkm_no_merge", "residual_hkm_no_merge_adaptive_t"),
]
for flag in (True, False):
    print(f"\n=== include_outliers={flag} ===")
    for fixed_m, adapt_m in pairs:
        f = fixed[(fixed.method == fixed_m) & (fixed.include_outliers == flag)]
        a = adaptive[(adaptive.method == adapt_m) & (adaptive.include_outliers == flag)]
        print(f"{fixed_m:>28}: ME {f.ME_percent.mean():6.2f} +/- {f.ME_percent.std():5.2f} | "
              f"CountAcc {f.CountAcc.mean():.3f} | AbsK {f.AbsK.mean():.3f}")
        print(f"{adapt_m:>28}: ME {a.ME_percent.mean():6.2f} +/- {a.ME_percent.std():5.2f} | "
              f"CountAcc {a.CountAcc.mean():.3f} | AbsK {a.AbsK.mean():.3f}")
        fa = f.groupby("scene_id").ME_percent.mean()
        aa = a.groupby("scene_id").ME_percent.mean()
        d = (aa - fa).dropna()
        wins = int((d < -1e-6).sum()); losses = int((d > 1e-6).sum())
        p = float("nan")
        if wilcoxon is not None and (np.abs(d) > 1e-6).any():
            p = float(wilcoxon(d[np.abs(d) > 1e-6]).pvalue)
        print(f"{'paired delta':>28}: mean {d.mean():+.2f} pp | wins {wins} / losses {losses} | wilcoxon p={p:.3f}")

# Per-scene view for the headline setting.
print("\n=== per-scene delta, HKM no_merge, include_outliers=True (adaptive - fixed) ===")
f = fixed[(fixed.method == "residual_hkm_no_merge") & (fixed.include_outliers == True)].groupby("scene_id")[["ME_percent", "K_gt", "K_est"]].mean()
a = adaptive[(adaptive.method == "residual_hkm_no_merge_adaptive_t") & (adaptive.include_outliers == True)].groupby("scene_id")[["ME_percent", "K_est", "sigma_hat"]].mean()
m = f.join(a, lsuffix="_fixed", rsuffix="_adapt")
m["delta"] = m.ME_percent_adapt - m.ME_percent_fixed
print(m[["sigma_hat", "K_gt", "K_est_fixed", "K_est_adapt", "ME_percent_fixed", "ME_percent_adapt", "delta"]]
      .sort_values("delta").round(2).to_string())
