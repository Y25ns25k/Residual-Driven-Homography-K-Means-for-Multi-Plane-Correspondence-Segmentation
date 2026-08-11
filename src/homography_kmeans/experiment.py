from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except Exception:  # pragma: no cover - scipy is an optional runtime fallback.
    scipy_wilcoxon = None

from .adelaide import AdelaideScene, filter_adelaide_scenes, load_adelaide_directory_report
from .energy import assign_by_residual, error_matrix, estimate_scales
from .hkm import ResidualHomographyKMeans
from .metrics import best_label_mapping_and_correct_mask, evaluate_segmentation
from .ransac import estimate_homography_ransac
from .sequential import sequential_ransac
from .spatial import icm_smooth_labels, knn_edges
from .synthetic import SyntheticScene, generate_synthetic_suite
from .visualization import save_correspondence_plot, save_k_error_hist, save_residual_histogram, save_threshold_curve


LOGGER = logging.getLogger(__name__)

SUMMARY_METRICS = [
    "ME",
    "ME_percent",
    "SegAcc",
    "CountAcc",
    "AbsK",
    "OverSeg",
    "UnderSeg",
    "Runtime",
]

V2_SEED_SOURCE = "residual_hkm_no_merge"
V2_LAMBDA_S = 0.5


def _seed_method_name(method: str) -> str:
    """Keep v2 paired to the audited no-merge initialization seed."""
    return V2_SEED_SOURCE if method == "residual_hkm_v2" else method


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_dir(config: dict[str, Any], run_name: str | None = None) -> Path:
    output_root = Path(config.get("output_root", "outputs"))
    name = run_name or config.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root / str(name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "labels").mkdir(exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    return path


def _setup_log(run_dir: Path) -> None:
    handlers: list[logging.Handler] = [logging.FileHandler(run_dir / "logs.txt", encoding="utf-8"), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)


def _save_run_config(config: dict[str, Any], run_dir: Path) -> None:
    with (run_dir / "config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def _method_seed(base_seed: int, scene_index: int, method: str, setting_offset: int = 0) -> int:
    stable_method_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(method))
    return int(base_seed + 1009 * scene_index + 37 * stable_method_offset + setting_offset)


def _adelaide_method_seed(base_seed: int, scene_id: str, method: str, setting_offset: int = 0) -> int:
    stable_scene_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(scene_id.lower()))
    stable_method_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(method))
    return int(base_seed + 1009 * stable_scene_offset + 37 * stable_method_offset + setting_offset)


def _method_fit(scene: SyntheticScene | AdelaideScene, method: str, config: dict[str, Any], seed: int):
    x1 = scene.x1
    x2 = scene.x2
    image_shape = getattr(scene, "image_shape", None) or tuple(config.get("image_shape", [480, 640]))
    rcfg = config.get("ransac", {})
    hcfg = config.get("hkm", {})

    if method in {"global_ransac", "single_homography_ransac"}:
        t0 = time.perf_counter()
        result = estimate_homography_ransac(
            x1,
            x2,
            threshold=float(rcfg.get("threshold", 3.0)),
            max_iterations=int(rcfg.get("max_iterations", 1500)),
            confidence=float(rcfg.get("confidence", 0.999)),
            min_support=int(rcfg.get("min_support", hcfg.get("min_support", 20))),
            random_state=seed,
            refine=True,
        )
        runtime = time.perf_counter() - t0
        labels = np.full(len(x1), -1, dtype=np.int32)
        homographies: list[np.ndarray] = []
        if result.success and result.homography is not None:
            homographies = [result.homography]
            labels[result.inlier_mask] = 0
        diagnostics = {
            "alias": "single_homography_ransac",
            "n_iter": float(result.n_iter),
            "n_inliers": float(result.n_inliers),
            "success": float(result.success),
            **result.diagnostics,
        }
        return homographies, labels, result.residuals.copy(), runtime, diagnostics

    if method in {"sequential_ransac", "seq_global_reassignment"}:
        t0 = time.perf_counter()
        seq = sequential_ransac(
            x1,
            x2,
            threshold=float(rcfg.get("threshold", 3.0)),
            max_iterations=int(rcfg.get("max_iterations", 1500)),
            confidence=float(rcfg.get("confidence", 0.999)),
            min_support=int(rcfg.get("min_support", hcfg.get("min_support", 20))),
            max_models=rcfg.get("max_models", None),
            random_state=seed,
        )
        runtime = time.perf_counter() - t0
        labels = seq.labels.copy()
        scales = np.ones(len(seq.homographies), dtype=np.float64)
        residuals = seq.residuals.copy()
        if method == "seq_global_reassignment" and seq.homographies:
            errors = error_matrix(seq.homographies, x1, x2)
            scales = estimate_scales(seq.homographies, labels, x1, x2, float(hcfg.get("sigma_min", 0.75)))
            labels = assign_by_residual(
                errors,
                scales,
                float(hcfg.get("tau_abs", 4.5)),
                float(hcfg.get("tau_norm", 3.0)),
                scale_adaptive=True,
            )
            residuals = np.full(len(x1), np.inf)
            valid = labels >= 0
            residuals[np.flatnonzero(valid)] = errors[np.flatnonzero(valid), labels[valid]]
        return seq.homographies, labels, residuals, runtime, {"scales": scales.tolist()}

    toggles = {
        "hkm_no_discovery": dict(use_residual_discovery=False, use_functional_merge=False, use_energy_merge=False),
        "residual_hkm_no_merge": dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False),
        "residual_hkm_v2": dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False),
        "residual_hkm_conservative": dict(
            use_residual_discovery=True,
            use_functional_merge=False,
            use_energy_merge=False,
            use_conservative_assignment=True,
            use_conservative_discovery=True,
        ),
        "residual_hkm_functional_merge": dict(use_residual_discovery=True, use_functional_merge=True, use_energy_merge=False),
        "residual_hkm_energy_merge": dict(use_residual_discovery=True, use_functional_merge=True, use_energy_merge=True),
        "residual_hkm_no_scale_adaptive": dict(
            use_residual_discovery=True,
            use_functional_merge=True,
            use_energy_merge=True,
            use_scale_adaptive=False,
        ),
    }
    if method not in toggles:
        raise ValueError(f"unknown method: {method}")
    use_v2 = method == "residual_hkm_v2"
    km = ResidualHomographyKMeans(config, random_state=seed, use_rank4_prior=use_v2, **toggles[method])
    result = km.fit(x1, x2, image_shape=image_shape)
    if use_v2 and result.homographies:
        t0 = time.perf_counter()
        errors = error_matrix(result.homographies, x1, x2)
        scales = estimate_scales(result.homographies, result.labels, x1, x2, km._energy_config().sigma_min)
        neighbors = knn_edges(x1, k=8)
        labels = icm_smooth_labels(errors, scales, result.labels, neighbors, km._energy_config(), lambda_s=V2_LAMBDA_S)
        diagnostics = dict(result.diagnostics)
        diagnostics["v2_rank4_prior"] = True
        diagnostics["v2_icm_lambda_s"] = V2_LAMBDA_S
        diagnostics["v2_icm_runtime"] = float(time.perf_counter() - t0)
        return result.homographies, labels, result.residuals, result.runtime + diagnostics["v2_icm_runtime"], diagnostics
    return result.homographies, result.labels, result.residuals, result.runtime, result.diagnostics


def _evaluate_rows(scenes, config: dict[str, Any], methods: list[str], seed: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    first_residuals: dict[str, np.ndarray] = {}
    include_outliers = bool(config.get("include_outliers", False))
    for si, scene in enumerate(scenes):
        gt_h = getattr(scene, "gt_homographies", None)
        image_shape = getattr(scene, "image_shape", None) or tuple(config.get("image_shape", [480, 640]))
        for method in methods:
            method_seed = _method_seed(seed, si, _seed_method_name(method))
            Hs, labels, residuals, runtime, diagnostics = _method_fit(scene, method, config, method_seed)
            metrics = evaluate_segmentation(
                scene.gt_labels if hasattr(scene, "gt_labels") else scene.labels,
                labels,
                pred_homographies=Hs,
                x1=scene.x1,
                x2=scene.x2,
                gt_homographies=gt_h,
                image_shape=image_shape,
                include_outliers=include_outliers,
                runtime=runtime,
            )
            metrics["ME_percent"] = 100.0 * metrics["ME"]
            row = {
                "scene_id": scene.scene_id,
                "difficulty": getattr(scene, "difficulty", "adelaide"),
                "method": method,
                **metrics,
            }
            rows.append(row)
            if si == 0:
                first_residuals[method] = residuals
            yield_labels = np.asarray(labels, dtype=np.int32)
            diagnostics["n_labels"] = int(len(yield_labels))
            row["diagnostics_json"] = json.dumps(diagnostics, default=float)
            yield_label_path = row.get("_label_path")
            _ = yield_label_path
    return rows, first_residuals


def _adelaide_seed_values(config: dict[str, Any]) -> list[int]:
    base_seed = int(config.get("seed", 42))
    explicit = config.get("seeds")
    if explicit:
        values = [int(v) for v in explicit]
    else:
        n = max(5, int(config.get("num_seeds", 5)))
        values = [base_seed + 100003 * i for i in range(n)]
    while len(values) < 5:
        values.append(base_seed + 100003 * len(values))
    return values


def _include_outlier_values(config: dict[str, Any]) -> list[bool]:
    if "include_outliers_values" in config:
        raw = config.get("include_outliers_values") or []
        values = [bool(v) for v in raw]
    else:
        values = [False, True]
    if False not in values:
        values.insert(0, False)
    if True not in values:
        values.append(True)
    return values


def _evaluate_adelaide_rows_multi_policy(
    scenes: list[AdelaideScene],
    config: dict[str, Any],
    methods: list[str],
    seed: int,
    seed_index: int,
    include_outliers_values: list[bool],
    label_dir: Path | None = None,
    label_include_outliers: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    first_residuals: dict[str, np.ndarray] = {}
    for si, scene in enumerate(scenes):
        gt_h = getattr(scene, "gt_homographies", None)
        image_shape = scene.image_shape or tuple(config.get("image_shape", [480, 640]))
        for method in methods:
            method_seed = _adelaide_method_seed(seed, scene.scene_id, _seed_method_name(method))
            Hs, labels, residuals, runtime, diagnostics = _method_fit(scene, method, config, method_seed)
            if si == 0:
                first_residuals[method] = residuals
            diagnostics = dict(diagnostics)
            diagnostics["n_labels"] = int(len(labels))
            diagnostics_json = json.dumps(diagnostics, default=float)
            label_payload_metrics: dict[str, float] | None = None
            for include_outliers in include_outliers_values:
                metrics = evaluate_segmentation(
                    scene.labels,
                    labels,
                    pred_homographies=Hs,
                    x1=scene.x1,
                    x2=scene.x2,
                    gt_homographies=gt_h,
                    image_shape=image_shape,
                    include_outliers=include_outliers,
                    runtime=runtime,
                )
                metrics["ME_percent"] = 100.0 * metrics["ME"]
                if include_outliers == label_include_outliers or label_payload_metrics is None:
                    label_payload_metrics = metrics
                rows.append(
                    {
                        "scene_id": scene.scene_id,
                        "difficulty": "adelaide",
                        "method": method,
                        "seed": int(seed),
                        "seed_index": int(seed_index),
                        "include_outliers": bool(include_outliers),
                        **metrics,
                        "diagnostics_json": diagnostics_json,
                    }
                )
            if label_dir is not None and label_payload_metrics is not None:
                mapping, correct_mask = best_label_mapping_and_correct_mask(
                    scene.labels,
                    labels,
                    include_outliers=label_include_outliers,
                )
                mapping_pairs = np.asarray(sorted(mapping.items()), dtype=np.int32)
                np.savez_compressed(
                    label_dir / f"{scene.scene_id}_seed{int(seed)}_{method}.npz",
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
                    K_gt=np.asarray(label_payload_metrics["K_gt"], dtype=np.float64),
                    K_est=np.asarray(label_payload_metrics["K_est"], dtype=np.float64),
                    ME=np.asarray(label_payload_metrics["ME"], dtype=np.float64),
                    SegAcc=np.asarray(label_payload_metrics["SegAcc"], dtype=np.float64),
                    CountAcc=np.asarray(label_payload_metrics["CountAcc"], dtype=np.float64),
                    AbsK=np.asarray(label_payload_metrics["AbsK"], dtype=np.float64),
                    include_outliers=np.asarray(bool(label_include_outliers)),
                )
    return rows, first_residuals


def _write_labels(run_dir: Path, scenes, config: dict[str, Any], methods: list[str], seed: int) -> None:
    for si, scene in enumerate(scenes):
        for method in methods:
            Hs, labels, residuals, runtime, diagnostics = _method_fit(
                scene,
                method,
                config,
                _method_seed(seed, si, _seed_method_name(method)),
            )
            np.save(run_dir / "labels" / f"{scene.scene_id}_{method}.npy", labels.astype(np.int32))


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"status": "empty"}
    metric_cols = [
        "K_est",
        "AbsK",
        "CountAcc",
        "OverSeg",
        "UnderSeg",
        "SegAcc",
        "ME",
        "ME_percent",
        "OutlierPrecision",
        "OutlierRecall",
        "OutlierF1",
        "MedianTransferErr",
        "MedianCornerErr",
        "Runtime",
    ]
    out: dict[str, Any] = {"status": "ok", "n_rows": int(len(df)), "methods": {}}
    for method, sub in df.groupby("method"):
        vals = {}
        for col in metric_cols:
            if col in sub:
                vals[col] = float(pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean())
        out["methods"][method] = vals
    return out


def _summary_stats(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    groups = group_cols or ["method"]
    available = [c for c in SUMMARY_METRICS if c in df.columns]
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(groups, keys)}
        for metric in available:
            vals = pd.to_numeric(sub[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.count() > 1 else 0.0
        row["n"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_me_delta_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute scene-paired ME deltas against Sequential RANSAC.

    Returns an aggregate table and a per-scene table. Positive delta means the
    method has higher ME than Sequential RANSAC on the same scene and threshold.
    """
    required = {"threshold_setting", "scene_id", "method", "ME"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()

    per_scene_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    eps = 1e-6
    for threshold_setting, sub in df.groupby("threshold_setting", dropna=False):
        seq = sub[sub["method"] == "sequential_ransac"][["scene_id", "ME"]].rename(columns={"ME": "ME_sequential"})
        if seq.empty:
            continue
        for method, msub in sub.groupby("method", dropna=False):
            merged = msub[["scene_id", "ME"]].merge(seq, on="scene_id", how="inner")
            if merged.empty:
                continue
            deltas = (
                pd.to_numeric(merged["ME"], errors="coerce")
                - pd.to_numeric(merged["ME_sequential"], errors="coerce")
            ).replace([np.inf, -np.inf], np.nan)
            valid = deltas.dropna().to_numpy(dtype=np.float64)
            for scene_id, me_method, me_seq, delta in zip(
                merged["scene_id"].tolist(),
                merged["ME"].tolist(),
                merged["ME_sequential"].tolist(),
                deltas.tolist(),
            ):
                per_scene_rows.append(
                    {
                        "threshold_setting": threshold_setting,
                        "method": method,
                        "scene_id": scene_id,
                        "ME_method": float(me_method),
                        "ME_sequential": float(me_seq),
                        "delta_ME": float(delta),
                    }
                )
            if len(valid):
                win_count = int(np.sum(valid < -eps))
                loss_count = int(np.sum(valid > eps))
                tie_count = int(np.sum(np.abs(valid) <= eps))
                stat = float("nan")
                pvalue = float("nan")
                if method == "sequential_ransac":
                    stat = 0.0
                    pvalue = 1.0
                elif scipy_wilcoxon is not None and len(valid) > 0:
                    nonzero = valid[np.abs(valid) > eps]
                    if len(nonzero) == 0:
                        stat = 0.0
                        pvalue = 1.0
                    else:
                        try:
                            res = scipy_wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
                            stat = float(res.statistic)
                            pvalue = float(res.pvalue)
                        except ValueError:
                            stat = float("nan")
                            pvalue = float("nan")
                aggregate_rows.append(
                    {
                        "threshold_setting": threshold_setting,
                        "method": method,
                        "delta_ME_mean": float(np.mean(valid)),
                        "delta_ME_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                        "win_count": win_count,
                        "loss_count": loss_count,
                        "tie_count": tie_count,
                        "wilcoxon_stat": stat,
                        "wilcoxon_pvalue": pvalue,
                        "n": int(len(valid)),
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_scene_rows)


def write_reports(summary: dict[str, Any], run_dir: Path, report_dir: str | Path = "report") -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = summary.get("methods", {})
    lines = [
        "# Scale-Adaptive Residual-Driven Homography K-Means",
        "",
        "## 1. Problem formulation",
        "Given point correspondences between two images, the task is to estimate several plane-specific homographies and assign each correspondence to one plane or to an outlier label.",
        "",
        "## 2. Motivation",
        "A single homography is insufficient when correspondences come from multiple planes. Sequential RANSAC can initialize several homographies, but its greedy removal step can lock in early assignment errors.",
        "",
        "## 3. Baseline",
        "The baseline is greedy Sequential RANSAC: repeatedly fit one homography to remaining correspondences, remove its inliers, and stop at the configured support/model limits.",
        "",
        "## 4. Proposed method",
        "Sequential RANSAC gives greedy initial homographies; residual-driven Homography K-Means reassigns/refits all correspondences, discovers missed planes from structured residuals, and merges duplicate homographies using functional/energy criteria.",
        "",
        "## 5. Mathematical details",
        "The implementation uses non-squared symmetric transfer error, MAD residual scales, Huber-weighted normalized DLT refits, outlier-pool residual discovery, functional warp distance, and an energy with data, model-count, and outlier penalties.",
        "",
        "## 6. Experiments and results",
        f"Fresh outputs were read from `{run_dir.as_posix()}`. Metrics below are recomputed outputs only.",
    ]
    if methods:
        lines.append("")
        lines.append("| method | ME | SegAcc | CountAcc | AbsK | Runtime |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method, vals in methods.items():
            lines.append(
                f"| {method} | {vals.get('ME', float('nan')):.3f} | {vals.get('SegAcc', float('nan')):.3f} | "
                f"{vals.get('CountAcc', float('nan')):.3f} | {vals.get('AbsK', float('nan')):.3f} | {vals.get('Runtime', float('nan')):.3f} |"
            )
    lines.extend(
        [
            "",
            "## 7. Limitations",
            "K estimation remains difficult. Real correspondences can cause oversegmentation or undersegmentation, and thresholds affect the trade-off. Physical/rank-4 consistency is left as future work, not the main method.",
            "",
            "## 8. Conclusion",
            "The method is a lightweight, interpretable refinement over Sequential RANSAC. It should be claimed as an empirical correspondence-labeling refinement when validated, not as a state-of-the-art multi-model fitting result.",
        ]
    )
    (out / "project_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    story = [
        "# 발표용 최종 스토리",
        "",
        "이 프로젝트는 수업에서 다룬 feature matching, homography, RANSAC, clustering을 기반으로 한 고전 컴퓨터 비전 파이프라인입니다.",
        "",
        "핵심 아이디어는 Sequential RANSAC의 greedy assignment를 그대로 믿지 않고, residual-driven Homography K-Means로 모든 correspondence를 다시 할당하고 다시 fitting하는 것입니다.",
        "",
        "강하게 주장할 수 있는 부분은 재현 실험에서 Sequential RANSAC 대비 correspondence-level segmentation이 개선되는지입니다. 반대로 plane 개수 K를 항상 정확히 찾는다고 주장하면 안 됩니다.",
        "",
        "약한 부분은 K estimation trade-off입니다. threshold가 낮으면 oversegmentation이 생기고, 높으면 undersegmentation이 생길 수 있습니다.",
        "",
        "금지할 주장은 SOTA multi-model fitting을 이겼다거나, 항상 정확한 plane 개수를 찾는다는 식의 과장입니다.",
    ]
    (out / "final_story.md").write_text("\n".join(story) + "\n", encoding="utf-8")


def run_synthetic_experiment(config_path: str | Path, run_name: str | None = None) -> Path:
    config = load_config(config_path)
    run_dir = _run_dir(config, run_name)
    _setup_log(run_dir)
    _save_run_config(config, run_dir)
    seed = int(config.get("seed", 42))
    scenes = generate_synthetic_suite(config, seed=seed)
    methods = list(config.get("methods", ["sequential_ransac", "residual_hkm_energy_merge"]))
    LOGGER.info("running synthetic experiment: %d scenes, %d methods", len(scenes), len(methods))
    rows, first_residuals = _evaluate_rows(scenes, config, methods, seed)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    df.to_csv(run_dir / "per_scene_results.csv", index=False)
    _summary_stats(df).to_csv(run_dir / "summary_stats.csv", index=False)
    _write_labels(run_dir, scenes, config, methods, seed)
    summary = _summary(df)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figs = run_dir / "figures"
    if len(scenes):
        save_correspondence_plot(scenes[0].x1, scenes[0].x2, scenes[0].gt_labels, figs / "segmentation_examples.png", "ground truth labels")
    if not df.empty:
        primary = df[df["method"] == methods[-1]]
        save_k_error_hist(primary["AbsK"].to_numpy(), figs / "k_error_hist.png")
        save_threshold_curve(df.rename(columns={"method": "tau_abs"}), figs / "threshold_sweep_me.png", x_col="tau_abs")
    save_residual_histogram(first_residuals, figs / "residual_histograms.png")
    if bool(config.get("write_reports", True)):
        write_reports(summary, run_dir)
    return run_dir


def run_threshold_sweep(config_path: str | Path, run_name: str | None = None) -> Path:
    config = load_config(config_path)
    sweep = config.get("threshold_sweep", {})
    run_dir = _run_dir(config, run_name or f"{config.get('run_name', 'synthetic')}_threshold_sweep")
    _setup_log(run_dir)
    _save_run_config(config, run_dir)
    seed = int(config.get("seed", 42))
    scenes = generate_synthetic_suite(config, seed=seed)
    rows: list[dict[str, Any]] = []
    tau_abs_values = sweep.get("tau_abs", [config.get("hkm", {}).get("tau_abs", 4.5)])
    tau_norm_values = sweep.get("tau_norm", [config.get("hkm", {}).get("tau_norm", 3.0)])
    for tau_abs in tau_abs_values:
        for tau_norm in tau_norm_values:
            cfg2 = json.loads(json.dumps(config))
            cfg2.setdefault("hkm", {})["tau_abs"] = float(tau_abs)
            cfg2.setdefault("hkm", {})["tau_norm"] = float(tau_norm)
            cfg2["methods"] = ["residual_hkm_energy_merge"]
            partial, _ = _evaluate_rows(scenes, cfg2, ["residual_hkm_energy_merge"], seed + int(100 * float(tau_abs) + 10 * float(tau_norm)))
            for row in partial:
                row["tau_abs"] = float(tau_abs)
                row["tau_norm"] = float(tau_norm)
            rows.extend(partial)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    df.to_csv(run_dir / "per_scene_results.csv", index=False)
    _summary_stats(df, ["method", "tau_abs", "tau_norm"]).to_csv(run_dir / "summary_stats.csv", index=False)
    summary = _summary(df)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figs = run_dir / "figures"
    if not df.empty:
        save_threshold_curve(df, figs / "threshold_sweep_me.png", x_col="tau_abs")
        save_k_error_hist(df["AbsK"].to_numpy(), figs / "k_error_hist.png")
    save_residual_histogram({}, figs / "residual_histograms.png")
    if len(scenes):
        save_correspondence_plot(scenes[0].x1, scenes[0].x2, scenes[0].gt_labels, figs / "segmentation_examples.png", "threshold sweep example")
    if bool(config.get("write_reports", True)):
        write_reports(summary, run_dir)
    return run_dir


def _calibration_settings(config: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    hkm = config.get("hkm", {})
    cal = config.get("merge_calibration", {})
    base = json.loads(json.dumps(config))
    settings: list[tuple[str, Any, dict[str, Any]]] = [("base", "base", base)]
    specs = [
        ("lambda_K", "lambda_K", hkm.get("lambda_K", 20.0)),
        ("gamma_outlier", "gamma_outlier", hkm.get("gamma_outlier", 8.0)),
        ("functional_merge_threshold", "merge_threshold", hkm.get("merge_threshold", 4.0)),
        ("min_support", "min_support", hkm.get("min_support", 20)),
    ]
    for public_name, hkm_name, default in specs:
        for value in cal.get(public_name, [default]):
            if str(value) == str(default):
                continue
            cfg = json.loads(json.dumps(config))
            cfg.setdefault("hkm", {})[hkm_name] = value
            settings.append((public_name, value, cfg))
    return settings


def run_merge_calibration(config_path: str | Path, run_name: str | None = None) -> Path:
    config = load_config(config_path)
    run_dir = _run_dir(config, run_name or f"{config.get('run_name', 'synthetic')}_merge_calibration")
    _setup_log(run_dir)
    _save_run_config(config, run_dir)
    seed = int(config.get("seed", 42))
    scenes = generate_synthetic_suite(config, seed=seed)
    methods = ["residual_hkm_no_merge", "residual_hkm_functional_merge", "residual_hkm_energy_merge"]
    rows: list[dict[str, Any]] = []
    for setting_idx, (param, value, cfg) in enumerate(_calibration_settings(config)):
        cfg["methods"] = methods
        partial, _ = _evaluate_rows(scenes, cfg, methods, seed + 7919 * setting_idx)
        hkm = cfg.get("hkm", {})
        for row in partial:
            row["sweep_param"] = param
            row["sweep_value"] = value
            row["lambda_K"] = float(hkm.get("lambda_K", config.get("hkm", {}).get("lambda_K", 20.0)))
            row["gamma_outlier"] = float(hkm.get("gamma_outlier", config.get("hkm", {}).get("gamma_outlier", 8.0)))
            row["functional_merge_threshold"] = float(hkm.get("merge_threshold", config.get("hkm", {}).get("merge_threshold", 4.0)))
            row["min_support"] = int(hkm.get("min_support", config.get("hkm", {}).get("min_support", 20)))
        rows.extend(partial)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    df.to_csv(run_dir / "per_scene_results.csv", index=False)
    stats = _summary_stats(df, ["sweep_param", "sweep_value", "method"])
    stats.to_csv(run_dir / "summary_stats.csv", index=False)
    stats.to_csv(run_dir / "merge_calibration_summary.csv", index=False)
    summary = _summary(df)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figs = run_dir / "figures"
    if not df.empty:
        primary = df[df["method"] == "residual_hkm_energy_merge"]
        save_k_error_hist(primary["AbsK"].to_numpy(), figs / "k_error_hist.png")
    if scenes:
        save_correspondence_plot(scenes[0].x1, scenes[0].x2, scenes[0].gt_labels, figs / "segmentation_examples.png", "merge calibration example")
    save_residual_histogram({}, figs / "residual_histograms.png")
    return run_dir


def run_conservative_calibration(config_path: str | Path, run_name: str | None = None) -> Path:
    config = load_config(config_path)
    run_dir = _run_dir(config, run_name or f"{config.get('run_name', 'synthetic')}_conservative_calibration")
    _setup_log(run_dir)
    _save_run_config(config, run_dir)
    seed = int(config.get("seed", 42))
    scenes = generate_synthetic_suite(config, seed=seed)
    cal = config.get("conservative_calibration", {})
    reassignment_values = cal.get("reassignment_margin", [config.get("hkm", {}).get("reassignment_margin", 0.15)])
    improvement_values = cal.get("discovery_improvement_margin", [config.get("hkm", {}).get("discovery_improvement_margin", 0.2)])
    coverage_values = cal.get("spatial_coverage_min", [config.get("hkm", {}).get("spatial_coverage_min", 0.05)])
    rows: list[dict[str, Any]] = []
    setting_idx = 0
    for reassignment_margin in reassignment_values:
        for improvement_margin in improvement_values:
            for coverage_min in coverage_values:
                cfg = json.loads(json.dumps(config))
                cfg["methods"] = ["residual_hkm_conservative"]
                cfg.setdefault("hkm", {})["reassignment_margin"] = float(reassignment_margin)
                cfg.setdefault("hkm", {})["discovery_improvement_margin"] = float(improvement_margin)
                cfg.setdefault("hkm", {})["spatial_coverage_min"] = float(coverage_min)
                partial, _ = _evaluate_rows(scenes, cfg, ["residual_hkm_conservative"], seed + 1291 * setting_idx)
                for row in partial:
                    row["reassignment_margin"] = float(reassignment_margin)
                    row["discovery_improvement_margin"] = float(improvement_margin)
                    row["spatial_coverage_min"] = float(coverage_min)
                rows.extend(partial)
                setting_idx += 1
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    df.to_csv(run_dir / "per_scene_results.csv", index=False)
    stats = _summary_stats(
        df,
        ["method", "reassignment_margin", "discovery_improvement_margin", "spatial_coverage_min"],
    )
    stats.to_csv(run_dir / "summary_stats.csv", index=False)
    stats.to_csv(run_dir / "conservative_calibration_summary.csv", index=False)
    summary = _summary(df)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figs = run_dir / "figures"
    if not df.empty:
        save_k_error_hist(df["AbsK"].to_numpy(), figs / "k_error_hist.png")
    if scenes:
        save_correspondence_plot(scenes[0].x1, scenes[0].x2, scenes[0].gt_labels, figs / "segmentation_examples.png", "conservative calibration example")
    save_residual_histogram({}, figs / "residual_histograms.png")
    return run_dir


def run_adelaide_experiment(
    config_path: str | Path,
    data_path: str | Path,
    run_name: str | None = None,
    subset: str = "all",
) -> Path:
    config = load_config(config_path)
    config["adelaide_subset"] = subset
    run_dir = _run_dir(config, run_name)
    _setup_log(run_dir)
    _save_run_config(config, run_dir)
    load_report = load_adelaide_directory_report(data_path)
    scenes, missing_subset_scenes = filter_adelaide_scenes(load_report.scenes, subset)
    if missing_subset_scenes:
        LOGGER.warning("missing %d configured Adelaide %s scenes: %s", len(missing_subset_scenes), subset, missing_subset_scenes)
    LOGGER.info("Adelaide subset %s: selected %d of %d parsed scenes", subset, len(scenes), len(load_report.scenes))
    if not scenes:
        message = {
            "status": "skipped",
            "reason": f"No parseable AdelaideRMF .mat files found at {Path(data_path)} for subset {subset}.",
            "mat_files_found": load_report.files_found,
            "subset": subset,
            "evaluated_scenes": 0,
            "skipped_scenes": len(load_report.skipped),
            "missing_subset_scenes": missing_subset_scenes,
            "skipped_files": load_report.skipped,
        }
        pd.DataFrame([]).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame([]).to_csv(run_dir / "per_scene_results.csv", index=False)
        pd.DataFrame([]).to_csv(run_dir / "summary_stats.csv", index=False)
        (run_dir / "summary.json").write_text(json.dumps(message, indent=2), encoding="utf-8")
        (run_dir / "logs.txt").write_text(message["reason"] + "\n", encoding="utf-8")
        if bool(config.get("write_reports", True)):
            write_reports(message, run_dir)
        return run_dir
    methods = list(config.get("methods", ["sequential_ransac", "residual_hkm_energy_merge"]))
    seeds = _adelaide_seed_values(config)
    include_values = _include_outlier_values(config)
    rows: list[dict[str, Any]] = []
    first_residuals: dict[str, np.ndarray] = {}
    for seed_index, seed_value in enumerate(seeds):
        partial, residuals = _evaluate_adelaide_rows_multi_policy(
            scenes,
            config,
            methods,
            seed_value,
            seed_index,
            include_values,
            label_dir=run_dir / "labels",
            label_include_outliers=True,
        )
        for row in partial:
            row["subset"] = subset
        rows.extend(partial)
        if not first_residuals:
            first_residuals = residuals
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    df.to_csv(run_dir / "per_scene_results.csv", index=False)
    stats = _summary_stats(df, ["subset", "include_outliers", "method"])
    stats.to_csv(run_dir / "summary_stats.csv", index=False)
    paper_cols = [
        "subset",
        "include_outliers",
        "method",
        "ME_percent_mean",
        "ME_percent_std",
        "ME_mean",
        "ME_std",
        "SegAcc_mean",
        "CountAcc_mean",
        "AbsK_mean",
        "OverSeg_mean",
        "UnderSeg_mean",
        "Runtime_mean",
        "n",
    ]
    stats[[c for c in paper_cols if c in stats.columns]].to_csv(run_dir / "paper_summary.csv", index=False)
    summary = _summary(df)
    summary["subset"] = subset
    summary["seeds"] = seeds
    summary["include_outliers_values"] = include_values
    summary["mat_files_found"] = load_report.files_found
    summary["parsed_scenes"] = int(len(load_report.scenes))
    summary["evaluated_scenes"] = int(len(scenes))
    summary["skipped_scenes"] = int(len(load_report.skipped))
    summary["missing_subset_scenes"] = missing_subset_scenes
    summary["skipped_files"] = load_report.skipped
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figs = run_dir / "figures"
    if not df.empty:
        save_k_error_hist(df["AbsK"].to_numpy(), figs / "k_error_hist.png")
        save_threshold_curve(df.rename(columns={"method": "tau_abs"}), figs / "threshold_sweep_me.png", x_col="tau_abs")
    save_residual_histogram(first_residuals, figs / "residual_histograms.png")
    if scenes:
        save_correspondence_plot(scenes[0].x1, scenes[0].x2, scenes[0].labels, figs / "segmentation_examples.png", "Adelaide labels")
    if bool(config.get("write_reports", True)):
        write_reports(summary, run_dir)
    return run_dir


def run_adelaide_threshold_sweep(config_path: str | Path, data_path: str | Path, run_name: str | None = None) -> Path:
    config = load_config(config_path)
    run_dir = _run_dir(config, run_name or f"adelaide_threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    _setup_log(run_dir)
    _save_run_config(config, run_dir)
    load_report = load_adelaide_directory_report(data_path)
    scenes = load_report.scenes
    if not scenes:
        message = {
            "status": "skipped",
            "reason": f"No parseable AdelaideRMF .mat files found at {Path(data_path)}.",
            "mat_files_found": load_report.files_found,
            "evaluated_scenes": 0,
            "skipped_scenes": len(load_report.skipped),
            "skipped_files": load_report.skipped,
        }
        pd.DataFrame([]).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame([]).to_csv(run_dir / "summary_stats.csv", index=False)
        (run_dir / "summary.json").write_text(json.dumps(message, indent=2), encoding="utf-8")
        return run_dir

    settings = [
        ("strict", 2.0, 3.0, 2.5),
        ("base", 2.5, 4.0, 3.0),
        ("mid", 3.5, 5.5, 3.5),
        ("loose", 5.0, 7.0, 4.0),
        ("very_loose", 7.0, 9.0, 4.5),
    ]
    methods = list(
        config.get(
            "methods",
            [
                "sequential_ransac",
                "residual_hkm_no_merge",
                "residual_hkm_conservative",
                "residual_hkm_functional_merge",
                "residual_hkm_energy_merge",
            ],
        )
    )
    rows: list[dict[str, Any]] = []
    base_seed = int(config.get("seed", 42))
    for si, (setting, ransac_threshold, tau_abs, tau_norm) in enumerate(settings):
        cfg = json.loads(json.dumps(config))
        cfg.setdefault("ransac", {})["threshold"] = ransac_threshold
        cfg.setdefault("hkm", {})["tau_abs"] = tau_abs
        cfg.setdefault("hkm", {})["tau_norm"] = tau_norm
        partial, _ = _evaluate_rows(scenes, cfg, methods, base_seed + 4099 * si)
        for row in partial:
            row["threshold_setting"] = setting
            row["ransac_threshold"] = ransac_threshold
            row["tau_abs"] = tau_abs
            row["tau_norm"] = tau_norm
        rows.extend(partial)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    df.to_csv(run_dir / "per_scene_results.csv", index=False)
    stats = _summary_stats(df, ["threshold_setting", "method"])
    stats.to_csv(run_dir / "summary_stats.csv", index=False)
    requested_cols = [
        "threshold_setting",
        "method",
        "ME_mean",
        "ME_std",
        "SegAcc_mean",
        "CountAcc_mean",
        "AbsK_mean",
        "OverSeg_mean",
        "UnderSeg_mean",
        "Runtime_mean",
    ]
    stats[[c for c in requested_cols if c in stats.columns]].rename(
        columns={"threshold_setting": "threshold_name", "Runtime_mean": "runtime_mean"}
    ).to_csv(run_dir / "threshold_sweep_summary.csv", index=False)
    paired_stats, paired_per_scene = _paired_me_delta_stats(df)
    paired_stats.to_csv(run_dir / "paired_me_deltas.csv", index=False)
    paired_per_scene.to_csv(run_dir / "paired_me_deltas_per_scene.csv", index=False)
    summary = _summary(df)
    summary["mat_files_found"] = load_report.files_found
    summary["evaluated_scenes"] = int(len(scenes))
    summary["skipped_scenes"] = int(len(load_report.skipped))
    summary["skipped_files"] = load_report.skipped
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figs = run_dir / "figures"
    if not df.empty:
        save_threshold_curve(df, figs / "threshold_sweep_me.png", x_col="tau_abs")
        save_k_error_hist(df["AbsK"].to_numpy(), figs / "k_error_hist.png")
    if scenes:
        save_correspondence_plot(scenes[0].x1, scenes[0].x2, scenes[0].labels, figs / "segmentation_examples.png", "Adelaide threshold sweep example")
    return run_dir
