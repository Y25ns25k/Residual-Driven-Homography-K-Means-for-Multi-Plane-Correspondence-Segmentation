"""Audit step 5: failure analysis for scenes where HKM no-merge hurts ME.

Uses outputs/audit_ablation/metrics.csv (include_outliers=True), aggregates
per-scene over the 5 seeds, and inspects the scenes with the largest positive
paired delta ME_hkm_no_merge - ME_sequential.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.experiment import load_config

config = load_config(ROOT / "configs/adelaide.yml")
TAU_ABS = float(config["hkm"]["tau_abs"])
TAU_NORM = float(config["hkm"]["tau_norm"])
RANSAC_T = float(config["ransac"]["threshold"])

df = pd.read_csv(ROOT / "outputs/audit_ablation/metrics.csv")
df = df[df.include_outliers == True]

cols = ["ME_percent", "K_gt", "K_est", "SegAcc", "OutlierPrecision", "OutlierRecall"]
seq = df[df.method == "sequential_ransac"].groupby("scene_id")[cols].mean()
hkm = df[df.method == "residual_hkm_no_merge"].groupby("scene_id")[cols].mean()

merged = seq.join(hkm, lsuffix="_seq", rsuffix="_hkm")
merged["delta_ME"] = merged.ME_percent_hkm - merged.ME_percent_seq
fail = merged[merged.delta_ME > 0].sort_values("delta_ME", ascending=False)

print(f"thresholds used for ALL scenes (no per-scene tuning): "
      f"RANSAC inlier threshold = {RANSAC_T} px, tau_abs = {TAU_ABS} px, tau_norm = {TAU_NORM}")
print(f"\nscenes where HKM no-merge hurts (mean over 5 seeds): {len(fail)}")

rows = []
for scene_id, r in fail.head(5).iterrows():
    K_gt = r.K_gt_seq
    K_seq = r.K_est_seq
    K_hkm = r.K_est_hkm
    # Failure taxonomy:
    #  (a) wrong K: HKM's mean K differs from GT by >= 0.5
    #  (b) correct K, wrong assignment: K approx right, ME still worse
    #  (c) threshold mismatch indicator from GT-outlier handling:
    #      low OutlierRecall = GT outliers absorbed as inliers (too loose)
    #      low OutlierPrecision = real inliers expelled as outliers (too tight)
    if abs(K_hkm - K_gt) >= 0.5:
        cat = "(a) wrong K (" + ("under" if K_hkm < K_gt else "over") + "segmentation)"
    else:
        cat = "(b) correct K, wrong assignment"
    if r.OutlierRecall_hkm < 0.3:
        thr_note = f"threshold too loose for this scene (OutlierRecall={r.OutlierRecall_hkm:.2f})"
    elif r.OutlierPrecision_hkm < 0.3:
        thr_note = f"threshold too tight for this scene (OutlierPrecision={r.OutlierPrecision_hkm:.2f})"
    else:
        thr_note = f"threshold plausible (P={r.OutlierPrecision_hkm:.2f}, R={r.OutlierRecall_hkm:.2f})"
    rows.append(
        {
            "scene": scene_id,
            "K_gt": round(K_gt, 2),
            "K_seq": round(K_seq, 2),
            "K_hkm": round(K_hkm, 2),
            "ME_seq": round(r.ME_percent_seq, 2),
            "ME_hkm": round(r.ME_percent_hkm, 2),
            "delta_ME": round(r.delta_ME, 2),
            "tau_abs": TAU_ABS,
            "category": cat,
            "threshold_note": thr_note,
        }
    )

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(ROOT / "outputs/audit_ablation/failure_cases.csv", index=False)
print("\nwrote outputs/audit_ablation/failure_cases.csv")
