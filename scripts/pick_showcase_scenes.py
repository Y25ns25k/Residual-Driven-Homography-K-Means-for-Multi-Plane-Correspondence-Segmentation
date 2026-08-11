"""Pick showcase scenes for the final deck from the fairness-relaxed run.

Wanted: 2-3 good cases for v2+orphan7 vs sequential+orphan7 (at least one
where v2's K is closer to K_gt than seq's), and 1 failure case.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/fairness_relaxed").glob("metrics_*.csv"))],
    ignore_index=True,
)

seq = df[df.method == "sequential_ransac+orphan7.0"].groupby("scene_id")[["ME_percent", "K_gt", "K_est"]].mean()
v2 = df[df.method == "residual_hkm_v2+orphan7.0"].groupby("scene_id")[["ME_percent", "K_est"]].mean()
m = seq.join(v2, lsuffix="_seq", rsuffix="_v2")
m["delta"] = m.ME_percent_v2 - m.ME_percent_seq
m["dK_seq"] = (m.K_est_seq - m.K_gt).abs()
m["dK_v2"] = (m.K_est_v2 - m.K_gt).abs()
m["K_better"] = m.dK_v2 < m.dK_seq - 0.1
print(m[["K_gt", "K_est_seq", "K_est_v2", "ME_percent_seq", "ME_percent_v2", "delta", "K_better"]]
      .sort_values("delta").round(2).to_string())

print("\nper-seed detail for candidate scenes (to pick a representative seed):")
for scene in m.sort_values("delta").index[:6].tolist() + m.sort_values("delta").index[-3:].tolist():
    sub_s = df[(df.method == "sequential_ransac+orphan7.0") & (df.scene_id == scene)][["seed", "ME_percent", "K_est"]]
    sub_v = df[(df.method == "residual_hkm_v2+orphan7.0") & (df.scene_id == scene)][["seed", "ME_percent", "K_est"]]
    j = sub_s.merge(sub_v, on="seed", suffixes=("_seq", "_v2"))
    j["delta"] = j.ME_percent_v2 - j.ME_percent_seq
    print(f"\n{scene} (K_gt={int(m.loc[scene].K_gt)}):")
    print(j.round(2).to_string(index=False))
