"""Final analysis: 2x2 init/assign decoupling + post-hoc tau scan vs user sweep."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sweep = pd.read_csv(ROOT / "outputs/h_threshold_diagnostic_20260611_222500/metrics.csv")
sweep["ME_percent"] = 100.0 * sweep["ME"] if sweep["ME"].max() <= 1.0 else sweep["ME"]
ref = sweep.groupby("setting")[["ME_percent", "CountAcc", "AbsK", "OverSeg", "UnderSeg"]].mean()

dec = pd.concat(
    [pd.read_csv(p) for p in sorted((ROOT / "outputs/decoupled_thresholds").glob("metrics_*.csv"))],
    ignore_index=True,
)
new = dec.groupby("method")[["ME_percent", "CountAcc", "AbsK", "OverSeg", "UnderSeg"]].mean()

print("=== reference (user's coupled sweep, v2, include_outliers=true) ===")
order = ["base", "mid", "loose", "very_loose", "adaptive"]
print(ref.reindex(order).round(3).to_string())

print("\n=== decoupled variants (5 seeds x 19 scenes) ===")
print(new.round(3).to_string())

print("\n=== 2x2 view (ME / CountAcc / AbsK) ===")
cells = {
    "init=base,  assign=base  (v2 base)": ref.loc["base"],
    "init=base,  assign=loose": new.loc["base_init_loose_assign"] if "base_init_loose_assign" in new.index else None,
    "init=loose, assign=base": new.loc["loose_init_base_assign"] if "loose_init_base_assign" in new.index else None,
    "init=loose, assign=loose (v2 loose)": ref.loc["loose"],
}
for name, row in cells.items():
    if row is None:
        print(f"{name:>38}: (missing)")
    else:
        print(f"{name:>38}: ME {row.ME_percent:6.2f} | CountAcc {row.CountAcc:.3f} | AbsK {row.AbsK:.3f}")

print("\n=== post-hoc final-assignment tau scan (fit fully at base; K frozen) ===")
for m in sorted(x for x in new.index if x.startswith("posthoc")):
    row = new.loc[m]
    print(f"{m:>16}: ME {row.ME_percent:6.2f} | CountAcc {row.CountAcc:.3f} | AbsK {row.AbsK:.3f}")
