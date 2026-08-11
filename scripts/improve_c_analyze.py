"""Improvement C analysis: ICM smoothing vs base, paired per scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

ROOT = Path(__file__).resolve().parents[1]
icm = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/improve_c").glob("metrics_icm_seeds_*.csv"))],
    ignore_index=True,
)
fixed = pd.read_csv(ROOT / "outputs/audit_ablation/metrics.csv")
base = fixed[(fixed.method == "residual_hkm_no_merge")]

for flag in (True, False):
    print(f"=== include_outliers={flag} ===")
    b = base[base.include_outliers == flag]
    print(f"{'base no_merge':>22}: ME {b.ME_percent.mean():6.2f} +/- {b.ME_percent.std():5.2f} | "
          f"SegAcc {b.SegAcc.mean():.3f} | CountAcc {b.CountAcc.mean():.3f} | AbsK {b.AbsK.mean():.3f}")
    for lam in (0.25, 0.5, 1.0):
        s = icm[(icm.lambda_s == lam) & (icm.include_outliers == flag)]
        bb = b.groupby("scene_id").ME_percent.mean()
        ss = s.groupby("scene_id").ME_percent.mean()
        d = (ss - bb).dropna()
        wins = int((d < -1e-6).sum()); losses = int((d > 1e-6).sum())
        p = float("nan")
        if wilcoxon is not None and (np.abs(d) > 1e-6).any():
            p = float(wilcoxon(d[np.abs(d) > 1e-6]).pvalue)
        print(f"{'+ ICM lam=' + str(lam):>22}: ME {s.ME_percent.mean():6.2f} +/- {s.ME_percent.std():5.2f} | "
              f"SegAcc {s.SegAcc.mean():.3f} | CountAcc {s.CountAcc.mean():.3f} | AbsK {s.AbsK.mean():.3f} | "
              f"paired {d.mean():+.2f} pp, W{wins}/L{losses}, p={p:.3f}")
    print()

print("=== per-scene delta, lam=0.5, include_outliers=True ===")
b = base[base.include_outliers == True].groupby("scene_id").ME_percent.mean()
s = icm[(icm.lambda_s == 0.5) & (icm.include_outliers == True)].groupby("scene_id").ME_percent.mean()
d = (s - b).sort_values()
print(d.round(2).to_string())
