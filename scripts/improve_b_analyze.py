"""Improvement B analysis: guided discovery (+ICM) vs base, paired per scene."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

ROOT = Path(__file__).resolve().parents[1]
guided = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/improve_b").glob("metrics_guided_seeds_*.csv"))],
    ignore_index=True,
)
fixed = pd.read_csv(ROOT / "outputs/audit_ablation/metrics.csv")
base = fixed[fixed.method == "residual_hkm_no_merge"]
seq = fixed[fixed.method == "sequential_ransac"]

for flag in (True, False):
    print(f"=== include_outliers={flag} ===")
    s0 = seq[seq.include_outliers == flag]
    b = base[base.include_outliers == flag]
    print(f"{'sequential_ransac':>32}: ME {s0.ME_percent.mean():6.2f} +/- {s0.ME_percent.std():5.2f} | "
          f"SegAcc {s0.SegAcc.mean():.3f} | CountAcc {s0.CountAcc.mean():.3f} | AbsK {s0.AbsK.mean():.3f}")
    print(f"{'no_merge (base)':>32}: ME {b.ME_percent.mean():6.2f} +/- {b.ME_percent.std():5.2f} | "
          f"SegAcc {b.SegAcc.mean():.3f} | CountAcc {b.CountAcc.mean():.3f} | AbsK {b.AbsK.mean():.3f}")
    for m in ("residual_hkm_no_merge_guided", "residual_hkm_no_merge_guided_icm"):
        g = guided[(guided.method == m) & (guided.include_outliers == flag)]
        bb = b.groupby("scene_id").ME_percent.mean()
        gg = g.groupby("scene_id").ME_percent.mean()
        d = (gg - bb).dropna()
        wins = int((d < -1e-6).sum()); losses = int((d > 1e-6).sum())
        p = float("nan")
        if wilcoxon is not None and (np.abs(d) > 1e-6).any():
            p = float(wilcoxon(d[np.abs(d) > 1e-6]).pvalue)
        print(f"{m:>32}: ME {g.ME_percent.mean():6.2f} +/- {g.ME_percent.std():5.2f} | "
              f"SegAcc {g.SegAcc.mean():.3f} | CountAcc {g.CountAcc.mean():.3f} | AbsK {g.AbsK.mean():.3f} | "
              f"Runtime {g.Runtime.mean():.2f}s | paired {d.mean():+.2f} pp, W{wins}/L{losses}, p={p:.3f}")
    print()

print("=== per-scene delta, guided vs base, include_outliers=True ===")
b = base[base.include_outliers == True].groupby("scene_id")[["ME_percent", "K_gt", "K_est"]].mean()
g = guided[(guided.method == "residual_hkm_no_merge_guided") & (guided.include_outliers == True)]
acc = g.groupby("scene_id").discovery_accepted.mean()
g2 = g.groupby("scene_id")[["ME_percent", "K_est"]].mean()
m = b.join(g2, lsuffix="_base", rsuffix="_guided").join(acc)
m["delta"] = m.ME_percent_guided - m.ME_percent_base
print(m[["K_gt", "K_est_base", "K_est_guided", "discovery_accepted", "ME_percent_base", "ME_percent_guided", "delta"]]
      .sort_values("delta").round(2).to_string())
