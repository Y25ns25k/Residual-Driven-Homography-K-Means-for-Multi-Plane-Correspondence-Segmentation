from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.adelaide import _load_payload, filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.experiment import _adelaide_method_seed, _method_fit, _seed_method_name, load_config
from homography_kmeans.metrics import best_label_mapping_and_correct_mask, evaluate_segmentation


METHOD_TITLES = {
    "gt": "GT labels",
    "global_ransac": "Global RANSAC (single H)",
    "single_homography_ransac": "Global RANSAC (single H)",
    "sequential_ransac": "Sequential RANSAC",
    "residual_hkm_no_merge": "Residual HKM no_merge",
    "residual_hkm_conservative": "Residual HKM conservative",
    "residual_hkm_functional_merge": "Residual HKM functional_merge",
    "residual_hkm_energy_merge": "Residual HKM energy_merge",
    "residual_hkm_v2": "Residual HKM v2",
}


def _label_values(labels: np.ndarray) -> list[int]:
    return [int(v) for v in sorted(np.unique(labels)) if int(v) >= 0]


def _k_count(labels: np.ndarray) -> int:
    return len(_label_values(labels))


def _scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def _float(value: Any) -> float:
    return float(_scalar(value))


def _int(value: Any) -> int:
    return int(round(float(_scalar(value))))


def _palette(label_values: list[int]) -> dict[int, Any]:
    cmap = plt.get_cmap("tab20")
    colors = {-1: (0.35, 0.35, 0.35, 1.0)}
    for i, label in enumerate(label_values):
        colors[int(label)] = cmap(i % 20)
    return colors


def _map_pred_labels(pred: np.ndarray, mapping_pairs: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    mapped = np.full(len(pred), -1, dtype=np.int32)
    pairs = np.asarray(mapping_pairs).reshape(-1, 2) if np.asarray(mapping_pairs).size else np.empty((0, 2), dtype=np.int32)
    mapping = {int(p): int(g) for p, g in pairs}
    next_label = max(_label_values(gt_labels) or [0]) + 1
    extra: dict[int, int] = {}
    for pred_label in sorted(np.unique(pred)):
        pred_i = int(pred_label)
        if pred_i < 0:
            mapped[pred == pred_i] = -1
        elif pred_i in mapping:
            mapped[pred == pred_i] = mapping[pred_i]
        else:
            if pred_i not in extra:
                extra[pred_i] = next_label
                next_label += 1
            mapped[pred == pred_i] = extra[pred_i]
    return mapped


def _colors_for(labels: np.ndarray, colors: dict[int, Any]) -> list[Any]:
    fallback = (0.1, 0.1, 0.1, 1.0)
    return [colors.get(int(v), fallback) for v in labels]


def _scene_paths(data_dir: Path) -> dict[str, Path]:
    return {p.stem.lower(): p for p in data_dir.rglob("*.mat")}


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _paired_seed_table(metrics: pd.DataFrame, baseline: str, method: str, include_outliers: bool) -> pd.DataFrame:
    df = metrics[_bool_series(metrics["include_outliers"]) == bool(include_outliers)].copy()
    df = df[df["method"].isin([baseline, method])]
    values = ["ME", "ME_percent", "SegAcc", "K_est", "K_gt", "CountAcc", "AbsK"]
    pivot = df.pivot_table(
        index=["scene_id", "seed", "seed_index"],
        columns="method",
        values=values,
        aggfunc="first",
    )
    pivot.columns = ["_".join([str(v) for v in col if str(v)]) for col in pivot.columns]
    pivot = pivot.reset_index()
    required = [f"ME_{baseline}", f"ME_{method}"]
    missing = [col for col in required if col not in pivot.columns]
    if missing:
        raise ValueError(f"missing method columns in results CSV: {missing}")
    pivot["delta_ME"] = pivot[f"ME_{method}"] - pivot[f"ME_{baseline}"]
    pivot["delta_ME_percent_point"] = pivot[f"ME_percent_{method}"] - pivot[f"ME_percent_{baseline}"]
    return pivot


def _select_cases(metrics: pd.DataFrame, baseline: str, method: str, include_outliers: bool) -> dict[str, dict[str, Any]]:
    pivot = _paired_seed_table(metrics, baseline, method, include_outliers)
    scene_means = pivot.groupby("scene_id", as_index=False)["delta_ME"].mean()
    scene_means = scene_means.rename(columns={"delta_ME": "delta_ME_mean"})
    best_scene = scene_means.sort_values("delta_ME_mean").iloc[0]
    failure_scene = scene_means.sort_values("delta_ME_mean").iloc[-1]
    median_delta = float(scene_means["delta_ME_mean"].median())
    median_scene = scene_means.iloc[(scene_means["delta_ME_mean"] - median_delta).abs().argsort().iloc[0]]

    out: dict[str, dict[str, Any]] = {}
    for case_name, row in [
        ("best_case", best_scene),
        ("median_case", median_scene),
        ("failure_case", failure_scene),
    ]:
        scene_id = str(row["scene_id"])
        candidates = pivot[pivot["scene_id"] == scene_id].copy()
        candidates["seed_distance"] = (candidates["delta_ME"] - float(row["delta_ME_mean"])).abs()
        chosen = candidates.sort_values(["seed_distance", "seed_index"]).iloc[0]
        out[case_name] = _row_from_pivot(chosen, baseline, method, float(row["delta_ME_mean"]))
    return out


def _row_from_pivot(row: pd.Series, baseline: str, method: str, delta_mean: float) -> dict[str, Any]:
    delta_pp = float(row[f"ME_percent_{method}"] - row[f"ME_percent_{baseline}"])
    if delta_pp < -1e-6:
        outcome = "improved"
    elif delta_pp > 1e-6:
        outcome = "worsened"
    else:
        outcome = "tied"
    return {
        "scene_id": str(row["scene_id"]),
        "seed": int(row["seed"]),
        "seed_index": int(row["seed_index"]),
        "delta_ME_mean": float(delta_mean),
        "delta_ME_seed": float(row["delta_ME"]),
        "delta_ME_percent_point": delta_pp,
        "ME_seq": float(row[f"ME_{baseline}"]),
        "ME_hkm": float(row[f"ME_{method}"]),
        "ME_seq_percent": float(row[f"ME_percent_{baseline}"]),
        "ME_hkm_percent": float(row[f"ME_percent_{method}"]),
        "SegAcc_seq": float(row[f"SegAcc_{baseline}"]),
        "SegAcc_hkm": float(row[f"SegAcc_{method}"]),
        "K_gt": int(round(float(row[f"K_gt_{baseline}"]))),
        "K_seq": int(round(float(row[f"K_est_{baseline}"]))),
        "K_hkm": int(round(float(row[f"K_est_{method}"]))),
        "CountAcc_seq": float(row[f"CountAcc_{baseline}"]),
        "CountAcc_hkm": float(row[f"CountAcc_{method}"]),
        "AbsK_seq": float(row[f"AbsK_{baseline}"]),
        "AbsK_hkm": float(row[f"AbsK_{method}"]),
        "outcome": outcome,
    }


def _select_all_scenes(
    metrics: pd.DataFrame,
    scene_ids: list[str],
    baseline: str,
    method: str,
    include_outliers: bool,
    representative_seed: str,
) -> list[dict[str, Any]]:
    if representative_seed != "scene_mean_delta":
        raise ValueError(f"unsupported representative seed strategy: {representative_seed}")
    pivot = _paired_seed_table(metrics, baseline, method, include_outliers)
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        candidates = pivot[pivot["scene_id"].astype(str).str.lower() == scene_id.lower()].copy()
        if candidates.empty:
            print(f"[visuals] no paired result rows for scene {scene_id}; skipping")
            continue
        delta_mean = float(candidates["delta_ME"].mean())
        candidates["seed_distance"] = (candidates["delta_ME"] - delta_mean).abs()
        chosen = candidates.sort_values(["seed_distance", "seed_index"]).iloc[0]
        rows.append(_row_from_pivot(chosen, baseline, method, delta_mean))
    rows.sort(key=lambda r: (r["delta_ME_percent_point"], r["scene_id"]))
    return rows


def _npz_path(results_dir: Path, scene_id: str, seed: int, method: str) -> Path:
    return results_dir / "labels" / f"{scene_id}_seed{int(seed)}_{method}.npz"


def _save_label_npz(
    path: Path,
    scene: Any,
    seed: int,
    method: str,
    config: dict[str, Any],
    include_outliers: bool,
) -> dict[str, Any]:
    method_seed = _adelaide_method_seed(seed, scene.scene_id, _seed_method_name(method))
    Hs, labels, residuals, runtime, diagnostics = _method_fit(scene, method, config, method_seed)
    metrics = evaluate_segmentation(
        scene.labels,
        labels,
        pred_homographies=Hs,
        x1=scene.x1,
        x2=scene.x2,
        image_shape=scene.image_shape or tuple(config.get("image_shape", [480, 640])),
        include_outliers=include_outliers,
        runtime=runtime,
    )
    mapping, correct_mask = best_label_mapping_and_correct_mask(scene.labels, labels, include_outliers=include_outliers)
    mapping_pairs = np.asarray(sorted(mapping.items()), dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scene_name=np.asarray(scene.scene_id),
        seed=np.asarray(int(seed), dtype=np.int64),
        method=np.asarray(method),
        x1=scene.x1,
        x2=scene.x2,
        gt_labels=scene.labels,
        pred_labels=np.asarray(labels, dtype=np.int32),
        label_mapping=mapping_pairs,
        correct_mask=correct_mask,
        residuals=np.asarray(residuals, dtype=np.float64),
        K_gt=np.asarray(metrics["K_gt"], dtype=np.float64),
        K_est=np.asarray(metrics["K_est"], dtype=np.float64),
        ME=np.asarray(metrics["ME"], dtype=np.float64),
        SegAcc=np.asarray(metrics["SegAcc"], dtype=np.float64),
        CountAcc=np.asarray(metrics["CountAcc"], dtype=np.float64),
        AbsK=np.asarray(metrics["AbsK"], dtype=np.float64),
        include_outliers=np.asarray(bool(include_outliers)),
        diagnostics_json=np.asarray(json.dumps(diagnostics, sort_keys=True)),
    )
    return dict(np.load(path, allow_pickle=False))


def _load_or_make_npz(
    results_dir: Path,
    scene: Any,
    seed: int,
    method: str,
    config: dict[str, Any],
    include_outliers: bool,
) -> dict[str, Any]:
    path = _npz_path(results_dir, scene.scene_id, seed, method)
    if not path.exists():
        return _save_label_npz(path, scene, seed, method, config, include_outliers)
    return dict(np.load(path, allow_pickle=False))


def _plot_label_panel(
    ax: plt.Axes,
    x1: np.ndarray,
    display_labels: np.ndarray,
    colors: dict[int, Any],
    title: str,
    correct: np.ndarray | None = None,
    point_size: float = 12.0,
) -> None:
    ax.scatter(x1[:, 0], x1[:, 1], s=point_size, c=_colors_for(display_labels, colors), linewidths=0.0, alpha=0.9)
    if correct is not None:
        wrong = ~np.asarray(correct, dtype=bool)
        if np.any(wrong):
            ax.scatter(
                x1[wrong, 0],
                x1[wrong, 1],
                s=max(point_size + 4.0, 8.0),
                marker="x",
                c="red",
                linewidths=0.7,
                alpha=0.9,
            )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def _case_payload(
    scene: Any,
    seed: int,
    baseline: str,
    method: str,
    results_dir: Path,
    config: dict[str, Any],
    include_outliers: bool,
) -> dict[str, Any]:
    seq = _load_or_make_npz(results_dir, scene, seed, baseline, config, include_outliers)
    hkm = _load_or_make_npz(results_dir, scene, seed, method, config, include_outliers)
    seq_disp = _map_pred_labels(seq["pred_labels"], seq["label_mapping"], scene.labels)
    hkm_disp = _map_pred_labels(hkm["pred_labels"], hkm["label_mapping"], scene.labels)
    all_labels = sorted(set(_label_values(scene.labels)) | set(_label_values(seq_disp)) | set(_label_values(hkm_disp)))
    colors = _palette(all_labels)
    return {"seq": seq, "hkm": hkm, "seq_disp": seq_disp, "hkm_disp": hkm_disp, "colors": colors}


def _save_label_grid(scene: Any, payload: dict[str, Any], out_dir: Path, baseline: str, method: str) -> Path:
    seq = payload["seq"]
    hkm = payload["hkm"]
    colors = payload["colors"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    _plot_label_panel(
        axes[0],
        scene.x1,
        scene.labels,
        colors,
        f"{scene.scene_id}\nGT labels\nK_gt={_k_count(scene.labels)}",
    )
    _plot_label_panel(
        axes[1],
        scene.x1,
        payload["seq_disp"],
        colors,
        f"{METHOD_TITLES.get(baseline, baseline)}\nME={100*_float(seq['ME']):.1f}%, K_gt={_k_count(scene.labels)}, K_est={_int(seq['K_est'])}",
        seq["correct_mask"],
    )
    _plot_label_panel(
        axes[2],
        scene.x1,
        payload["hkm_disp"],
        colors,
        f"{METHOD_TITLES.get(method, method)}\nME={100*_float(hkm['ME']):.1f}%, K_gt={_k_count(scene.labels)}, K_est={_int(hkm['K_est'])}",
        hkm["correct_mask"],
    )
    path = out_dir / "label_grid.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_case_figure(case_name: str, scene: Any, payload: dict[str, Any], out_dir: Path, baseline: str, method: str, prefix: str) -> Path:
    path = _save_label_grid(scene, payload, out_dir, baseline, method)
    target = out_dir / f"{prefix}_{case_name.replace('_case', '')}_case.png"
    path.replace(target)
    return target


def _load_images(scene_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    payload = _load_payload(scene_path)
    if "img1" not in payload or "img2" not in payload:
        return None
    return np.asarray(payload["img1"]), np.asarray(payload["img2"])


def _draw_match_row(
    ax: plt.Axes,
    img1: np.ndarray,
    img2: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    display_labels: np.ndarray,
    colors: dict[int, Any],
    title: str,
    rng: np.random.Generator,
    max_lines: int,
) -> None:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    if img1.ndim == 2:
        canvas = np.zeros((canvas_h, canvas_w), dtype=img1.dtype)
    else:
        channels = img1.shape[2]
        canvas = np.zeros((canvas_h, canvas_w, channels), dtype=img1.dtype)
    canvas[:h1, :w1, ...] = img1
    canvas[:h2, w1 : w1 + w2, ...] = img2
    ax.imshow(canvas, cmap="gray" if canvas.ndim == 2 else None)
    n = len(x1)
    idx = np.arange(n)
    if n > max_lines:
        idx = rng.choice(idx, size=max_lines, replace=False)
    for i in idx:
        color = colors.get(int(display_labels[i]), (0.1, 0.1, 0.1, 1.0))
        ax.plot([x1[i, 0], x2[i, 0] + w1], [x1[i, 1], x2[i, 1]], color=color, linewidth=0.6, alpha=0.55)
    ax.scatter(x1[idx, 0], x1[idx, 1], s=5, c=_colors_for(display_labels[idx], colors))
    ax.scatter(x2[idx, 0] + w1, x2[idx, 1], s=5, c=_colors_for(display_labels[idx], colors))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _save_match_overlay(
    scene: Any,
    scene_path: Path,
    payload: dict[str, Any],
    out_dir: Path,
    baseline: str,
    method: str,
    max_correspondences: int,
    filename: str = "match_overlay.png",
) -> Path | None:
    images = _load_images(scene_path)
    if images is None:
        return None
    img1, img2 = images
    seq = payload["seq"]
    hkm = payload["hkm"]
    colors = payload["colors"]
    rng = np.random.default_rng(1000 + sum(ord(c) for c in scene.scene_id))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    _draw_match_row(
        axes[0],
        img1,
        img2,
        scene.x1,
        scene.x2,
        scene.labels,
        colors,
        f"{scene.scene_id} - GT correspondences, K_gt={_k_count(scene.labels)}",
        rng,
        max_correspondences,
    )
    _draw_match_row(
        axes[1],
        img1,
        img2,
        scene.x1,
        scene.x2,
        payload["seq_disp"],
        colors,
        f"{METHOD_TITLES.get(baseline, baseline)}, ME={100*_float(seq['ME']):.1f}%, K_est={_int(seq['K_est'])}",
        rng,
        max_correspondences,
    )
    _draw_match_row(
        axes[2],
        img1,
        img2,
        scene.x1,
        scene.x2,
        payload["hkm_disp"],
        colors,
        f"{METHOD_TITLES.get(method, method)}, ME={100*_float(hkm['ME']):.1f}%, K_est={_int(hkm['K_est'])}",
        rng,
        max_correspondences,
    )
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_match_overlay_legacy(
    case_name: str,
    scene: Any,
    scene_path: Path,
    payload: dict[str, Any],
    out_dir: Path,
    baseline: str,
    method: str,
    prefix: str,
    max_correspondences: int,
) -> Path | None:
    return _save_match_overlay(
        scene,
        scene_path,
        payload,
        out_dir,
        baseline,
        method,
        max_correspondences,
        filename=f"{prefix}_{case_name.replace('_case', '')}_case_matches.png",
    )


def _save_grid(cases: dict[str, dict[str, Any]], scene_by_id: dict[str, Any], payloads: dict[str, dict[str, Any]], out_dir: Path, baseline: str, method: str, prefix: str) -> Path:
    row_names = ["best_case", "median_case", "failure_case"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for r, case_name in enumerate(row_names):
        scene = scene_by_id[cases[case_name]["scene_id"]]
        payload = payloads[case_name]
        seq = payload["seq"]
        hkm = payload["hkm"]
        colors = payload["colors"]
        _plot_label_panel(axes[r, 0], scene.x1, scene.labels, colors, f"{case_name.replace('_', ' ')}\nGT K={_k_count(scene.labels)}")
        _plot_label_panel(
            axes[r, 1],
            scene.x1,
            payload["seq_disp"],
            colors,
            f"Seq ME={100*_float(seq['ME']):.1f}% K={_int(seq['K_est'])}",
            seq["correct_mask"],
        )
        _plot_label_panel(
            axes[r, 2],
            scene.x1,
            payload["hkm_disp"],
            colors,
            f"HKM ME={100*_float(hkm['ME']):.1f}% K={_int(hkm['K_est'])}",
            hkm["correct_mask"],
        )
    path = out_dir / f"{prefix}_qualitative_grid.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_error_map(scene: Any, payload: dict[str, Any], out_dir: Path, filename: str = "error_map.png") -> Path:
    seq = payload["seq"]
    hkm = payload["hkm"]
    delta = 100.0 * (_float(hkm["ME"]) - _float(seq["ME"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, name, item in [(axes[0], "Sequential RANSAC", seq), (axes[1], "Residual HKM no_merge", hkm)]:
        gt_out = scene.labels < 0
        correct = np.asarray(item["correct_mask"], dtype=bool)
        colors = np.full((len(scene.x1), 4), [0.9, 0.1, 0.1, 1.0])
        colors[correct] = [0.1, 0.6, 0.2, 1.0]
        colors[gt_out] = [0.55, 0.55, 0.55, 1.0]
        ax.scatter(scene.x1[:, 0], scene.x1[:, 1], s=13, c=colors, linewidths=0.0)
        ax.set_title(
            f"{name}\nME_seq={100*_float(seq['ME']):.1f}% ME_hkm={100*_float(hkm['ME']):.1f}% delta={delta:+.1f}pp",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    path = out_dir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_error_map_legacy(case_name: str, scene: Any, payload: dict[str, Any], out_dir: Path, prefix: str) -> Path:
    return _save_error_map(scene, payload, out_dir, filename=f"{prefix}_error_map_{case_name.replace('_case', '')}.png")


def _save_k_bar(scene: Any, payload: dict[str, Any], out_dir: Path) -> Path:
    labels = ["K_gt", "K_seq", "K_hkm"]
    values = [_k_count(scene.labels), _int(payload["seq"]["K_est"]), _int(payload["hkm"]["K_est"])]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Number of non-outlier models")
    ax.set_title(f"{scene.scene_id}: K comparison")
    ax.set_ylim(0, max(values) + 1)
    path = out_dir / "k_bar.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_k_error(cases: dict[str, dict[str, Any]], out_dir: Path, prefix: str) -> Path:
    labels = ["best", "median", "failure"]
    x = np.arange(len(labels))
    width = 0.25
    k_gt = [cases[f"{name}_case"]["K_gt"] for name in labels]
    k_seq = [cases[f"{name}_case"]["K_seq"] for name in labels]
    k_hkm = [cases[f"{name}_case"]["K_hkm"] for name in labels]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(x - width, k_gt, width, label="K_gt", color="#4C78A8")
    ax.bar(x, k_seq, width, label="K_seq", color="#F58518")
    ax.bar(x + width, k_hkm, width, label="K_hkm", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of non-outlier models")
    ax.set_title("K estimation on selected AdelaideRMF-H scenes")
    ax.legend()
    path = out_dir / f"{prefix}_k_error_examples.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _scene_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": row["scene_id"],
        "selected_seed": row["seed"],
        "ME_seq": row["ME_seq"],
        "ME_hkm": row["ME_hkm"],
        "delta_ME": row["ME_hkm"] - row["ME_seq"],
        "ME_seq_percent": row["ME_seq_percent"],
        "ME_hkm_percent": row["ME_hkm_percent"],
        "delta_ME_percent_point": row["delta_ME_percent_point"],
        "SegAcc_seq": row["SegAcc_seq"],
        "SegAcc_hkm": row["SegAcc_hkm"],
        "K_gt": row["K_gt"],
        "K_seq": row["K_seq"],
        "K_hkm": row["K_hkm"],
        "CountAcc_seq": row["CountAcc_seq"],
        "CountAcc_hkm": row["CountAcc_hkm"],
        "AbsK_seq": row["AbsK_seq"],
        "AbsK_hkm": row["AbsK_hkm"],
        "outcome": row["outcome"],
    }


def _summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "scene": row["scene_id"],
                "selected_seed": row["seed"],
                "ME_seq_percent": row["ME_seq_percent"],
                "ME_hkm_percent": row["ME_hkm_percent"],
                "delta_ME_percent_point": row["delta_ME_percent_point"],
                "SegAcc_seq": row["SegAcc_seq"],
                "SegAcc_hkm": row["SegAcc_hkm"],
                "K_gt": row["K_gt"],
                "K_seq": row["K_seq"],
                "K_hkm": row["K_hkm"],
                "AbsK_seq": row["AbsK_seq"],
                "AbsK_hkm": row["AbsK_hkm"],
                "outcome": row["outcome"],
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["delta_ME_percent_point", "scene"]).reset_index(drop=True)


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_all_label_contact(rows: list[dict[str, Any]], scene_by_id: dict[str, Any], payloads: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11, max(1.4 * n, 8)), constrained_layout=True)
    if n == 1:
        axes = np.asarray([axes])
    for r, row in enumerate(rows):
        scene = scene_by_id[row["scene_id"]]
        payload = payloads[row["scene_id"]]
        colors = payload["colors"]
        seq = payload["seq"]
        hkm = payload["hkm"]
        prefix = f"{row['scene_id']} ({row['delta_ME_percent_point']:+.1f}pp)"
        _plot_label_panel(axes[r, 0], scene.x1, scene.labels, colors, f"{prefix}\nGT K={row['K_gt']}", point_size=5)
        _plot_label_panel(
            axes[r, 1],
            scene.x1,
            payload["seq_disp"],
            colors,
            f"Seq ME={100*_float(seq['ME']):.1f}% K={_int(seq['K_est'])}",
            seq["correct_mask"],
            point_size=5,
        )
        _plot_label_panel(
            axes[r, 2],
            scene.x1,
            payload["hkm_disp"],
            colors,
            f"HKM ME={100*_float(hkm['ME']):.1f}% K={_int(hkm['K_est'])}",
            hkm["correct_mask"],
            point_size=5,
        )
    path = out_dir / "h_all_label_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_image_contact(rows: list[dict[str, Any]], per_scene_root: Path, out_dir: Path, source_name: str, output_name: str, title: str) -> Path:
    n = len(rows)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.0, rows_n * 3.2), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(rows_n, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for i, row in enumerate(rows):
        ax = axes_arr.ravel()[i]
        img_path = per_scene_root / row["scene_id"] / source_name
        if img_path.exists():
            ax.imshow(plt.imread(img_path))
        ax.set_title(f"{row['scene_id']} {row['delta_ME_percent_point']:+.1f}pp", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    path = out_dir / output_name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_delta_bar(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [row["scene_id"] for row in rows]
    vals = np.asarray([row["delta_ME_percent_point"] for row in rows], dtype=float)
    colors = np.where(vals < -1e-6, "#54A24B", np.where(vals > 1e-6, "#E45756", "#9D9DA1"))
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("delta_ME percentage points (HKM - Sequential)")
    ax.set_title("AdelaideRMF-H all 19 scenes: HKM no_merge delta_ME")
    path = out_dir / "h_all_delta_bar.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_k_comparison(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [row["scene_id"] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(x, [row["K_gt"] for row in rows], "o-", label="K_gt", color="#4C78A8")
    ax.plot(x, [row["K_seq"] for row in rows], "s-", label="K_seq", color="#F58518")
    ax.plot(x, [row["K_hkm"] for row in rows], "^-", label="K_hkm", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Number of non-outlier models")
    ax.set_title("AdelaideRMF-H K comparison across all scenes")
    ax.legend()
    path = out_dir / "h_all_k_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_slide_summary(rows: list[dict[str, Any]], per_scene_root: Path, out_dir: Path) -> Path:
    best = rows[0]
    failure = rows[-1]
    median_delta = float(np.median([row["delta_ME_percent_point"] for row in rows]))
    median = sorted(rows, key=lambda row: abs(row["delta_ME_percent_point"] - median_delta))[0]

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0])
    ax_bar = fig.add_subplot(grid[:, 0])
    vals = np.asarray([row["delta_ME_percent_point"] for row in rows], dtype=float)
    labels = [row["scene_id"] for row in rows]
    colors = np.where(vals < -1e-6, "#54A24B", np.where(vals > 1e-6, "#E45756", "#9D9DA1"))
    y = np.arange(len(rows))
    ax_bar.barh(y, vals, color=colors)
    ax_bar.axvline(0, color="#333333", linewidth=1)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("delta_ME pp (HKM - Seq)")
    ax_bar.set_title("All 19 H scenes sorted by delta_ME")

    for r, row in enumerate([best, median, failure]):
        ax = fig.add_subplot(grid[r, 1])
        img_path = per_scene_root / row["scene_id"] / "label_grid.png"
        if img_path.exists():
            ax.imshow(plt.imread(img_path))
        ax.set_title(f"{row['outcome']}: {row['scene_id']} ({row['delta_ME_percent_point']:+.1f}pp)", fontsize=10)
        ax.axis("off")

    fig.suptitle("AdelaideRMF-H qualitative summary: all 19 scenes", fontsize=15)
    path = out_dir / "h_all_slide_summary.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def _write_all_readme(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]], contact_files: list[Path]) -> None:
    best = rows[:5]
    failures = rows[-5:][::-1]
    lines = [
        "# AdelaideRMF-H All-Scene Qualitative Visuals",
        "",
        "The AdelaideRMF-H subset contains 19 homography scenes. Earlier qualitative figures showed only three representative scenes; this folder contains all-scene visual results.",
        "",
        f"Results folder: `{args.results}`",
        f"Dataset subset: `{args.subset}`",
        f"Baseline: `{args.baseline}`",
        f"Method: `{args.method}`",
        f"Selection mode: `{args.selection}`",
        f"Representative seed strategy: `{args.representative_seed}`",
        "",
        "For each scene, the selected seed is the seed whose paired delta `ME_method - ME_baseline` is closest to that scene's mean delta across seeds. Metrics use `include_outliers=true`.",
        "",
        "Colors are matched to GT labels with Hungarian label matching. GT outliers are gray. Red x marks indicate points misclassified under the displayed method.",
        "",
        "H-only is the main real homography benchmark. These figures are qualitative inspection assets, not a state-of-the-art or exact-K claim.",
        "",
        "## Top 5 HKM Improvements",
        "",
        "| Scene | Seed | ME_seq | ME_hkm | delta_ME |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['scene_id']} | {row['seed']} | {row['ME_seq_percent']:.2f}% | {row['ME_hkm_percent']:.2f}% | {row['delta_ME_percent_point']:.2f} pp |"
        )
    lines.extend(["", "## Top 5 HKM Failures", "", "| Scene | Seed | ME_seq | ME_hkm | delta_ME |", "|---|---:|---:|---:|---:|"])
    for row in failures:
        lines.append(
            f"| {row['scene_id']} | {row['seed']} | {row['ME_seq_percent']:.2f}% | {row['ME_hkm_percent']:.2f}% | {row['delta_ME_percent_point']:.2f} pp |"
        )
    lines.extend(["", "## Contact Sheets", ""])
    for path in contact_files:
        lines.append(f"- `{path.relative_to(out_dir)}`")
    lines.extend(
        [
            "",
            "Reproduce:",
            "",
            "```bash",
            f"python scripts/make_adelaide_visuals.py --data \"{args.data}\" --results {args.results} --out {args.out} --subset {args.subset} --baseline {args.baseline} --method {args.method} --selection all --contact-sheet --max-correspondences {args.max_correspondences}",
            "```",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_representative_readme(out_dir: Path, args: argparse.Namespace, cases: dict[str, dict[str, Any]], overlays: list[Path]) -> None:
    title = "AdelaideRMF-H Qualitative Visuals" if args.subset == "homography" else f"AdelaideRMF-{args.subset} Diagnostic Qualitative Visuals"
    subset_note = (
        "These are paper-comparable homography-subset qualitative visuals."
        if args.subset == "homography"
        else "These are diagnostic visuals under a homography model; this subset is not the main homography benchmark."
    )
    lines = [
        f"# {title}",
        "",
        f"Results folder: `{args.results}`",
        f"Dataset subset: `{args.subset}`",
        f"Baseline: `{args.baseline}`",
        f"Method: `{args.method}`",
        "",
        subset_note,
        "",
        "Scenes were selected by seed-aggregated paired delta `ME_method - ME_baseline` using `include_outliers=true`.",
        "",
        "| Case | Scene | Seed | ME_seq | ME_hkm | delta_ME_mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case_name in ["best_case", "median_case", "failure_case"]:
        row = cases[case_name]
        lines.append(
            f"| {case_name} | {row['scene_id']} | {row['seed']} | {row['ME_seq_percent']:.2f}% | {row['ME_hkm_percent']:.2f}% | {100*row['delta_ME_mean']:.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"Image-pair overlays generated: `{bool(overlays)}`",
            "",
            "Reproduce:",
            "",
            "```bash",
            f"python scripts/make_adelaide_visuals.py --data \"{args.data}\" --results {args.results} --out {args.out} --subset {args.subset} --baseline {args.baseline} --method {args.method}",
            "```",
            "",
            "Warning: these qualitative visuals are for interpretation and presentation only. They are not a SOTA claim and do not imply exact plane-count recovery.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_representative(
    args: argparse.Namespace,
    data_dir: Path,
    results_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    scene_by_id: dict[str, Any],
    mat_paths: dict[str, Path],
) -> None:
    cases = _select_cases(metrics, args.baseline, args.method, include_outliers=True)
    generated: list[Path] = []
    overlays: list[Path] = []
    payloads: dict[str, dict[str, Any]] = {}
    for case_name, row in cases.items():
        scene = scene_by_id[row["scene_id"]]
        payload = _case_payload(scene, row["seed"], args.baseline, args.method, results_dir, config, include_outliers=True)
        payloads[case_name] = payload
        generated.append(_save_case_figure(case_name, scene, payload, out_dir, args.baseline, args.method, args.prefix))
        scene_path = mat_paths.get(scene.scene_id.lower())
        overlay = (
            _save_match_overlay_legacy(case_name, scene, scene_path, payload, out_dir, args.baseline, args.method, args.prefix, args.max_correspondences)
            if scene_path
            else None
        )
        if overlay is not None:
            overlays.append(overlay)
            generated.append(overlay)

    generated.append(_save_grid(cases, scene_by_id, payloads, out_dir, args.baseline, args.method, args.prefix))
    generated.append(_save_error_map_legacy("best_case", scene_by_id[cases["best_case"]["scene_id"]], payloads["best_case"], out_dir, args.prefix))
    generated.append(_save_error_map_legacy("failure_case", scene_by_id[cases["failure_case"]["scene_id"]], payloads["failure_case"], out_dir, args.prefix))
    generated.append(_save_k_error(cases, out_dir, args.prefix))
    _write_representative_readme(out_dir, args, cases, overlays)

    selected_path = out_dir / "selected_scenes.json"
    selected_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    generated.append(selected_path)

    print("[visuals] selected scenes:")
    for case_name, row in cases.items():
        print(
            f"  {case_name}: {row['scene_id']} seed={row['seed']} "
            f"ME_seq={row['ME_seq_percent']:.2f}% ME_hkm={row['ME_hkm_percent']:.2f}% "
            f"delta_mean={100*row['delta_ME_mean']:.2f}pp"
        )
    print("[visuals] generated files:")
    for path in generated:
        print(f"  {path}")


def _run_all(
    args: argparse.Namespace,
    results_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    scene_by_id: dict[str, Any],
    mat_paths: dict[str, Path],
    subset_scene_ids: list[str],
) -> None:
    rows = _select_all_scenes(
        metrics,
        subset_scene_ids,
        args.baseline,
        args.method,
        include_outliers=True,
        representative_seed=args.representative_seed,
    )
    per_scene_root = out_dir / "per_scene"
    tables_dir = out_dir / "tables"
    contact_dir = out_dir / "contact_sheets"
    per_scene_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if args.contact_sheet:
        contact_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict[str, Any]] = {}
    for row in rows:
        scene = scene_by_id[row["scene_id"]]
        scene_out = per_scene_root / row["scene_id"]
        scene_out.mkdir(parents=True, exist_ok=True)
        payload = _case_payload(scene, row["seed"], args.baseline, args.method, results_dir, config, include_outliers=True)
        payloads[row["scene_id"]] = payload
        _save_label_grid(scene, payload, scene_out, args.baseline, args.method)
        scene_path = mat_paths.get(scene.scene_id.lower())
        if scene_path is not None:
            _save_match_overlay(scene, scene_path, payload, scene_out, args.baseline, args.method, args.max_correspondences)
        _save_error_map(scene, payload, scene_out)
        _save_k_bar(scene, payload, scene_out)
        (scene_out / "summary.json").write_text(json.dumps(_scene_summary(row), indent=2), encoding="utf-8")
        print(
            f"[visuals] {row['scene_id']}: seed={row['seed']} "
            f"ME_seq={row['ME_seq_percent']:.2f}% ME_hkm={row['ME_hkm_percent']:.2f}% "
            f"delta={row['delta_ME_percent_point']:+.2f}pp"
        )

    summary_df = _summary_table(rows)
    summary_csv = tables_dir / "h_all_visual_summary.csv"
    summary_md = tables_dir / "h_all_visual_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    _write_markdown_table(summary_df, summary_md)

    contact_files: list[Path] = []
    if args.contact_sheet:
        contact_files.append(_save_all_label_contact(rows, scene_by_id, payloads, contact_dir))
        contact_files.append(
            _save_image_contact(
                rows,
                per_scene_root,
                contact_dir,
                "match_overlay.png",
                "h_all_match_grid.png",
                "AdelaideRMF-H compact match overlays",
            )
        )
        contact_files.append(
            _save_image_contact(
                rows,
                per_scene_root,
                contact_dir,
                "error_map.png",
                "h_all_error_grid.png",
                "AdelaideRMF-H compact error maps",
            )
        )
        contact_files.append(_save_delta_bar(rows, contact_dir))
        contact_files.append(_save_k_comparison(rows, contact_dir))
        contact_files.append(_save_slide_summary(rows, per_scene_root, contact_dir))

    _write_all_readme(out_dir, args, rows, contact_files)
    print(f"[visuals] visualized {len(rows)} scenes")
    print(f"[visuals] wrote summary tables to {tables_dir}")
    if contact_files:
        print("[visuals] contact sheets:")
        for path in contact_files:
            print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subset", choices=["all", "homography", "fundamental"], default="homography")
    parser.add_argument("--baseline", default="sequential_ransac")
    parser.add_argument("--method", default="residual_hkm_no_merge")
    parser.add_argument("--prefix", default="adelaide")
    parser.add_argument("--include-outliers", action="store_true", default=True)
    parser.add_argument("--selection", choices=["representative", "all"], default="representative")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--max-correspondences", type=int, default=150)
    parser.add_argument("--representative-seed", choices=["scene_mean_delta"], default="scene_mean_delta")
    args = parser.parse_args()

    data_dir = Path(args.data)
    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "labels").mkdir(parents=True, exist_ok=True)

    config = load_config(results_dir / "config.yml")
    metrics = pd.read_csv(results_dir / "per_scene_results.csv")

    load_report = load_adelaide_directory_report(data_dir)
    scenes, missing = filter_adelaide_scenes(load_report.scenes, args.subset)
    if missing:
        print(f"[visuals] missing configured scenes for subset {args.subset}: {missing}")
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    mat_paths = _scene_paths(data_dir)
    subset_scene_ids = [scene.scene_id for scene in scenes]

    if args.selection == "all":
        _run_all(args, results_dir, out_dir, config, metrics, scene_by_id, mat_paths, subset_scene_ids)
    else:
        _run_representative(args, data_dir, results_dir, out_dir, config, metrics, scene_by_id, mat_paths)


if __name__ == "__main__":
    main()
