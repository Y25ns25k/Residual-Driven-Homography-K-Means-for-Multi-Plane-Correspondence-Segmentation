"""Showcase visuals for the final deck: GT / Seq / HKM v2 under the final
relaxed-labeling convention (orphan recovery, tau=7) for hand-picked scenes.

Scenes and representative seeds were selected from the fairness run
(outputs/fairness_relaxed): bonhall (large ME gain, K equal), napierb
(K recovered 1->3), neem (K recovered 2->3), barrsmith (failure).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import _load_payload, filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.energy import error_matrix
from homography_kmeans.experiment import _adelaide_method_seed, _method_fit, load_config
from homography_kmeans.metrics import best_label_mapping_and_correct_mask, evaluate_segmentation

SHOWCASE = {
    "bonhall": 200129,
    "napierb": 100126,
    "neem": 300132,
    "barrsmith": 123,
}
TAU_F = 7.0

config = load_config(ROOT / "configs/adelaide.yml")
report = load_adelaide_directory_report(ROOT / "data/adelaidermf")
scenes = {s.scene_id.lower(): s for s in filter_adelaide_scenes(report.scenes, "homography")[0]}
mat_paths = {p.stem.lower(): p for p in (ROOT / "data/adelaidermf").rglob("*.mat")}
out_dir = ROOT / "presentation_assets" / "final_showcase"
out_dir.mkdir(parents=True, exist_ok=True)


def orphan_relabel(Hs, labels, x1, x2, tau=TAU_F):
    out = np.asarray(labels, dtype=np.int32).copy()
    if not Hs:
        return out
    orphans = np.flatnonzero(out < 0)
    if not len(orphans):
        return out
    errors = error_matrix(Hs, x1[orphans], x2[orphans])
    best = np.argmin(errors, axis=1)
    take = errors[np.arange(len(orphans)), best] <= tau
    out[orphans[take]] = best[take].astype(np.int32)
    return out


def to_gt_colors(gt_labels, pred_labels):
    """Map predicted labels onto GT ids (Hungarian) for consistent colors."""
    mapping, _ = best_label_mapping_and_correct_mask(gt_labels, pred_labels, include_outliers=True)
    out = np.full(len(pred_labels), -1, dtype=np.int32)
    next_id = int(max([v for v in gt_labels if v >= 0], default=0)) + 1
    extra = {}
    for p in sorted(np.unique(pred_labels)):
        p = int(p)
        if p < 0:
            continue
        if p in mapping and mapping[p] >= 0:
            out[pred_labels == p] = mapping[p]
        else:
            extra.setdefault(p, next_id + len(extra))
            out[pred_labels == p] = extra[p]
    return out


def load_images(scene_id):
    payload = _load_payload(mat_paths[scene_id])
    img1 = np.asarray(payload["img1"])
    img2 = np.asarray(payload["img2"])
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
    return img1, img2


PALETTE = plt.get_cmap("tab10")


def draw_row(ax, img1, img2, x1, x2, disp_labels, title, rng, max_lines=110):
    h = max(img1.shape[0], img2.shape[0])

    def pad(im):
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        if im.shape[0] < h:
            im = np.vstack([im, np.full((h - im.shape[0], im.shape[1], 3), 255, dtype=im.dtype)])
        return im

    img1, img2 = pad(img1), pad(img2)
    offset = img1.shape[1]
    ax.imshow(np.hstack([img1, img2]))
    colors = [(0.45, 0.45, 0.45, 1.0) if v < 0 else PALETTE(int(v) % 10) for v in disp_labels]
    ax.scatter(x1[:, 0], x1[:, 1], s=11, c=colors, linewidths=0)
    ax.scatter(x2[:, 0] + offset, x2[:, 1], s=11, c=colors, linewidths=0)
    idx = rng.choice(len(x1), size=min(max_lines, len(x1)), replace=False)
    for i in idx:
        ax.plot(
            [x1[i, 0], x2[i, 0] + offset], [x1[i, 1], x2[i, 1]],
            color=colors[i], linewidth=0.5, alpha=0.45,
        )
    ax.set_title(title, fontsize=13)
    ax.axis("off")


for scene_id, seed in SHOWCASE.items():
    scene = scenes[scene_id]
    x1, x2 = scene.x1, scene.x2
    img1, img2 = load_images(scene_id)
    rng = np.random.default_rng(7 + sum(ord(c) for c in scene_id))

    rows = [("GT", scene.labels, None)]
    for method, title in (("sequential_ransac", "Sequential RANSAC"), ("residual_hkm_v2", "Residual HKM v2")):
        mseed = _adelaide_method_seed(seed, scene.scene_id, method)
        Hs, labels, _, _, _ = _method_fit(scene, method, config, mseed)
        labels = orphan_relabel(Hs, np.asarray(labels, dtype=np.int32), x1, x2)
        m = evaluate_segmentation(scene.labels, labels, pred_homographies=Hs, x1=x1, x2=x2,
                                  include_outliers=True)
        rows.append((f"{title} (+orphan recovery), ME={100*m['ME']:.1f}%, K={int(m['K_est'])}", labels, m))

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), constrained_layout=True)
    draw_row(axes[0], img1, img2, x1, x2, scene.labels,
             f"{scene_id} - GT, K_gt={int(len({int(v) for v in scene.labels if v >= 0}))}", rng)
    for ax, (title, labels, _) in zip(axes[1:], rows[1:]):
        draw_row(ax, img1, img2, x1, x2, to_gt_colors(scene.labels, labels), title, rng)
    path = out_dir / f"{scene_id}_showcase.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path.name}: " + " | ".join(t for t, _, _ in rows[1:]))
