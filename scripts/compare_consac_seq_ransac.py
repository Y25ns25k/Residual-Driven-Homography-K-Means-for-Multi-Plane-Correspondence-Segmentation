"""Per-scene comparison: our Sequential RANSAC vs CONSAC paper's own
re-implementation (Kluger et al., CVPR 2020, supplemental Table 6)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# CONSAC supplemental Table 6, Sequential RANSAC column (their implementation,
# mean over five runs, ME %).
CONSAC_TABLE6_SEQ = {
    "barrsmith": 12.95,
    "bonhall": 20.43,
    "bonython": 0.00,
    "elderhalla": 16.36,
    "elderhallb": 18.67,
    "hartley": 9.38,
    "johnsona": 28.04,
    "johnsonb": 27.46,
    "ladysymon": 3.80,
    "library": 11.35,
    "napiera": 11.66,
    "napierb": 21.24,
    "neem": 14.44,
    "nese": 0.47,
    "oldclassicswing": 1.32,
    "physics": 0.00,
    "sene": 2.00,
    "unihouse": 10.69,
    "unionhouse": 1.51,
}

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "outputs/audit_ablation/metrics.csv")
ours = (
    df[(df.method == "sequential_ransac") & (df.include_outliers == True)]
    .groupby("scene_id")
    .ME_percent.mean()
)

rows = []
for scene, theirs in CONSAC_TABLE6_SEQ.items():
    o = float(ours.get(scene, float("nan")))
    rows.append({"scene": scene, "ours": round(o, 2), "consac_impl": theirs, "delta": round(o - theirs, 2)})
out = pd.DataFrame(rows).sort_values("delta", ascending=False)
print(out.to_string(index=False))
print(f"\nmean ours = {out.ours.mean():.2f}, mean CONSAC-impl = {out.consac_impl.mean():.2f}, "
      f"mean delta = {out.delta.mean():+.2f} pp")
better = int((out.delta < 0).sum())
worse = int((out.delta > 0).sum())
print(f"scenes where ours is better: {better}, worse: {worse}")
top = out.head(4)
print(f"top-4 gap scenes account for {top.delta.sum() / 19:.2f} pp of the {out.delta.mean():+.2f} pp mean gap")
