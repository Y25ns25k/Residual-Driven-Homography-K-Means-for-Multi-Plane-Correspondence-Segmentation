"""Per-scene analysis of the user's h_threshold_diagnostic sweep (v2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "outputs/h_threshold_diagnostic_20260611_222500/metrics.csv")
print("columns:", [c for c in df.columns][:20])
print("settings:", sorted(df.setting.unique()) if "setting" in df.columns else "?")

base = df[df.setting == "base"].groupby("scene_id")[["ME_percent", "K_gt", "K_est"]].mean()
loose = df[df.setting == "loose"].groupby("scene_id")[["ME_percent", "K_est"]].mean()
m = base.join(loose, lsuffix="_base", rsuffix="_loose")
m["dME"] = m.ME_percent_loose - m.ME_percent_base
m["dKerr"] = (m.K_est_loose - m.K_gt).abs() - (m.K_est_base - m.K_gt).abs()
print("\n=== per-scene: loose - base (v2), sorted by dME ===")
print(m[["K_gt", "K_est_base", "K_est_loose", "ME_percent_base", "ME_percent_loose", "dME", "dKerr"]]
      .sort_values("dME").round(2).to_string())
print(f"\nME gain total: {m.dME.mean():+.2f} pp | scenes where K error worsens: {(m.dKerr > 0.05).sum()}, "
      f"improves: {(m.dKerr < -0.05).sum()}")
print("\nscenes where ME improves AND K error worsens (the tradeoff cells):")
print(m[(m.dME < -0.5) & (m.dKerr > 0.05)][["K_gt", "K_est_base", "K_est_loose", "dME", "dKerr"]].round(2).to_string())
print("\nscenes where ME improves WITHOUT K damage:")
print(m[(m.dME < -0.5) & (m.dKerr <= 0.05)][["K_gt", "K_est_base", "K_est_loose", "dME", "dKerr"]].round(2).to_string())
