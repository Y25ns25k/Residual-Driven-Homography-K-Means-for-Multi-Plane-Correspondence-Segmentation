"""Audit step 4: rank-4 applicability analysis on AdelaideRMF-H fits.

Reads outputs/audit_ablation/rank4_singular_values.json and reports the K
distribution of the no-merge fits plus singular-value spectra for
representative scenes.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
log = json.loads((ROOT / "outputs/audit_ablation/rank4_singular_values.json").read_text(encoding="utf-8"))

ks = [entry["K"] for entry in log]
print("K distribution of residual_hkm_no_merge fits (scene x seed):")
for k, cnt in sorted(Counter(ks).items()):
    print(f"  K={k}: {cnt}/{len(ks)} ({100 * cnt / len(ks):.0f}%)")
applied = sum(1 for e in log if e["applied"])
print(f"rank-4 projection non-trivial (K>4): {applied}/{len(log)} fits")

# Representative scenes: largest K, a mid K, and a typical K=2 scene.
by_scene: dict[str, list[dict]] = {}
for e in log:
    by_scene.setdefault(e["scene_id"], []).append(e)

scene_maxk = sorted(by_scene.items(), key=lambda kv: -max(x["K"] for x in kv[1]))
chosen = []
seen_k = set()
for scene_id, entries in scene_maxk:
    best = max(entries, key=lambda x: x["K"])
    bucket = "high" if best["K"] >= 5 else ("mid" if best["K"] in (3, 4) else "low")
    if bucket not in seen_k:
        chosen.append((scene_id, best))
        seen_k.add(bucket)
    if len(chosen) == 3:
        break

print("\nsingular values of the stacked K x 9 homography matrix M:")
for scene_id, e in chosen:
    sv = np.asarray(e["singular_values"], dtype=np.float64)
    ratios = sv / sv[0] if len(sv) else sv
    print(f"\n  {scene_id} (seed {e['seed']}, K={e['K']}, rank-4 applied={e['applied']})")
    print(f"    sigma            : {np.array2string(sv, precision=4)}")
    print(f"    sigma / sigma_max: {np.array2string(ratios, precision=4)}")
    if e["K"] >= 5:
        tail = float(np.sqrt(np.sum(sv[4:] ** 2)) / np.sqrt(np.sum(sv**2)))
        print(f"    relative tail energy beyond rank 4: {tail:.4f}")
