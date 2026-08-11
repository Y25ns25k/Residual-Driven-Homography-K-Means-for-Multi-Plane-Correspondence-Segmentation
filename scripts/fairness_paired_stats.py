"""Paired stats for the fairness-relaxation table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
df = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/fairness_relaxed").glob("metrics_*.csv"))],
    ignore_index=True,
)

pairs = [
    ("sequential_ransac+orphan7.0", "global_ransac+orphan7.0"),
    ("residual_hkm_v2+orphan7.0", "sequential_ransac+orphan7.0"),
    ("residual_hkm_v2", "sequential_ransac"),
]
for a, b in pairs:
    da = df[df.method == a].set_index(["scene_id", "seed"]).ME_percent
    db = df[df.method == b].set_index(["scene_id", "seed"]).ME_percent
    d = (da - db).dropna()
    nz = d[np.abs(d) > 1e-6]
    p = float(wilcoxon(nz).pvalue) if len(nz) else float("nan")
    print(f"{a} - {b}: mean {d.mean():+.2f} pp | W{int((d < -1e-6).sum())}/L{int((d > 1e-6).sum())}/T{int((np.abs(d) <= 1e-6).sum())} | p={p:.2e}")
