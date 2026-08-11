from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.energy import assign_by_residual, error_matrix, estimate_scales
from homography_kmeans.experiment import _adelaide_method_seed, load_config
from homography_kmeans.hkm import ResidualHomographyKMeans
from homography_kmeans.metrics import evaluate_segmentation
from homography_kmeans.official_init import ConsacPayload, iter_consac_payloads, load_consac_payload
from homography_kmeans.sequential import sequential_ransac
from homography_kmeans.spatial import icm_smooth_labels, knn_edges


NO_MERGE = dict(use_residual_discovery=True, use_functional_merge=False, use_energy_merge=False)
LAMBDA_S = 0.5
SEED_VALUES = [123, 100126, 200129, 300132, 400135]


class OfficialInitHKM(ResidualHomographyKMeans):
    def __init__(self, initial_homographies: list[np.ndarray], initial_labels: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_homographies = [np.asarray(H, dtype=np.float64).copy() for H in initial_homographies]
        self.initial_labels = np.asarray(initial_labels, dtype=np.int32).copy()

    def _initialize(self, x1: np.ndarray, x2: np.ndarray):
        return [H.copy() for H in self.initial_homographies], self.initial_labels.copy()


def _initial_labels_from_homographies(payload: ConsacPayload, config: dict[str, Any]) -> np.ndarray:
    Hs = payload.homographies_pixel
    if not Hs:
        return np.full(len(payload.x1), -1, dtype=np.int32)
    hcfg = config.get("hkm", {})
    errors = error_matrix(Hs, payload.x1, payload.x2)
    scales = np.ones(len(Hs), dtype=np.float64)
    return assign_by_residual(
        errors,
        scales,
        float(hcfg.get("tau_abs", 4.0)),
        float(hcfg.get("tau_norm", 3.0)),
        scale_adaptive=False,
    )


def _seed_for(payload: ConsacPayload, method: str) -> int:
    base = SEED_VALUES[payload.run_index % len(SEED_VALUES)]
    return _adelaide_method_seed(base, payload.scene, method)


def _metrics_row(
    payload: ConsacPayload,
    method: str,
    labels: np.ndarray,
    homographies: list[np.ndarray],
    runtime: float,
    *,
    official_me: float | None = None,
    selected_instances: int | None = None,
) -> dict[str, Any]:
    metrics = evaluate_segmentation(
        payload.gt_labels,
        np.asarray(labels, dtype=np.int32),
        pred_homographies=homographies,
        x1=payload.x1,
        x2=payload.x2,
        image_shape=payload.image_shape,
        include_outliers=True,
        runtime=runtime,
    )
    if official_me is not None:
        metrics["ME"] = float(official_me)
        metrics["ME_percent"] = 100.0 * float(official_me)
        metrics["SegAcc"] = 1.0 - float(official_me)
    else:
        metrics["ME_percent"] = 100.0 * float(metrics["ME"])
    if selected_instances is not None:
        k_gt = int(metrics["K_gt"])
        k_est = int(selected_instances)
        metrics["K_est"] = float(k_est)
        metrics["AbsK"] = float(abs(k_est - k_gt))
        metrics["CountAcc"] = float(k_est == k_gt)
        metrics["OverSeg"] = float(k_est > k_gt)
        metrics["UnderSeg"] = float(k_est < k_gt)
    return {
        "scene_id": payload.scene,
        "run_index": int(payload.run_index),
        "method": method,
        **metrics,
        "Runtime": float(runtime),
    }


def _run_original_seq(payload: ConsacPayload, config: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
    rcfg = config.get("ransac", {})
    hcfg = config.get("hkm", {})
    t0 = time.perf_counter()
    seq = sequential_ransac(
        payload.x1,
        payload.x2,
        threshold=float(rcfg.get("threshold", 2.5)),
        max_iterations=int(rcfg.get("max_iterations", 2500)),
        confidence=float(rcfg.get("confidence", 0.999)),
        min_support=int(rcfg.get("min_support", hcfg.get("min_support", 20))),
        max_models=rcfg.get("max_models", 8),
        random_state=_seed_for(payload, "sequential_ransac"),
    )
    return seq.homographies, seq.labels, seq.residuals, time.perf_counter() - t0


def _run_hkm(payload: ConsacPayload, config: dict[str, Any], *, official_init: bool, v2: bool, method_name: str):
    seed = _seed_for(payload, method_name)
    init_labels = _initial_labels_from_homographies(payload, config) if official_init else None
    if official_init:
        km = OfficialInitHKM(
            payload.homographies_pixel,
            init_labels,
            config,
            random_state=seed,
            use_rank4_prior=v2,
            **NO_MERGE,
        )
    else:
        km = ResidualHomographyKMeans(config, random_state=seed, use_rank4_prior=v2, **NO_MERGE)
    fit = km.fit(payload.x1, payload.x2, image_shape=payload.image_shape)
    labels = fit.labels
    runtime = fit.runtime
    if v2 and fit.homographies:
        t0 = time.perf_counter()
        errors = error_matrix(fit.homographies, payload.x1, payload.x2)
        scales = estimate_scales(fit.homographies, fit.labels, payload.x1, payload.x2, km._energy_config().sigma_min)
        neighbors = knn_edges(payload.x1, k=8)
        labels = icm_smooth_labels(errors, scales, fit.labels, neighbors, km._energy_config(), lambda_s=LAMBDA_S)
        runtime += time.perf_counter() - t0
    return fit.homographies, np.asarray(labels, dtype=np.int32), fit.residuals, runtime


def _summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["ME_percent", "SegAcc", "CountAcc", "AbsK", "OverSeg", "UnderSeg", "Runtime"]
    rows = []
    for method, sub in df.groupby("method", dropna=False):
        row = {"method": method, "n": int(len(sub))}
        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.count() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.method == "official_uniform_only"][["scene_id", "run_index", "ME_percent"]].rename(
        columns={"ME_percent": "ME_official_uniform"}
    )
    rows = []
    for method, sub in df.groupby("method"):
        if method == "official_uniform_only":
            continue
        merged = sub[["scene_id", "run_index", "ME_percent"]].merge(base, on=["scene_id", "run_index"], how="inner")
        delta = merged["ME_percent"] - merged["ME_official_uniform"]
        wins = int((delta < -1e-6).sum())
        losses = int((delta > 1e-6).sum())
        ties = int((delta.abs() <= 1e-6).sum())
        stat = np.nan
        pvalue = np.nan
        nonzero = delta[np.abs(delta) > 1e-6].to_numpy(dtype=np.float64)
        if wilcoxon is not None and len(nonzero):
            res = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
            stat = float(res.statistic)
            pvalue = float(res.pvalue)
        rows.append(
            {
                "method": method,
                "delta_ME_percent_mean": float(delta.mean()),
                "delta_ME_percent_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "win_count": wins,
                "loss_count": losses,
                "tie_count": ties,
                "wilcoxon_stat": stat,
                "wilcoxon_pvalue": pvalue,
                "n": int(len(delta)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Accepted for command provenance; payloads contain the evaluated points.")
    parser.add_argument("--official", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", default="homography")
    parser.add_argument("--method", default="residual_hkm_v2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    official_dir = Path(args.official)
    out_dir = Path(args.out or f"outputs/hkm_official_init_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "command.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    runtime_lookup = {}
    per_scene_csv = official_dir / "official_uniform_per_scene.csv"
    if per_scene_csv.exists():
        official_rows = pd.read_csv(per_scene_csv)
        for _, row in official_rows.iterrows():
            runtime_lookup[(str(row["scene"]), int(row["run_index"]))] = float(row.get("runtime", np.nan))

    rows = []
    payload_paths = list(iter_consac_payloads(official_dir))
    for payload_path in payload_paths:
        payload = load_consac_payload(payload_path)
        official_runtime = runtime_lookup.get((payload.scene, payload.run_index), np.nan)
        rows.append(
            _metrics_row(
                payload,
                "official_uniform_only",
                payload.official_labels,
                payload.homographies_pixel,
                official_runtime,
                official_me=payload.official_miss_rate,
                selected_instances=payload.selected_instances,
            )
        )

        Hs, labels, _, runtime = _run_hkm(payload, config, official_init=True, v2=False, method_name="official_init_residual_hkm_no_merge")
        rows.append(_metrics_row(payload, "official_init_residual_hkm_no_merge", labels, Hs, runtime))

        Hs, labels, _, runtime = _run_hkm(payload, config, official_init=True, v2=True, method_name="official_init_residual_hkm_v2")
        rows.append(_metrics_row(payload, "official_init_residual_hkm_v2", labels, Hs, runtime))

        Hs, labels, _, runtime = _run_original_seq(payload, config)
        rows.append(_metrics_row(payload, "our_original_sequential_ransac", labels, Hs, runtime))

        Hs, labels, _, runtime = _run_hkm(payload, config, official_init=False, v2=True, method_name="our_original_residual_hkm_v2")
        rows.append(_metrics_row(payload, "our_original_residual_hkm_v2", labels, Hs, runtime))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    df.to_csv(out_dir / "per_payload_results.csv", index=False)
    stats = _summary_stats(df)
    stats.to_csv(out_dir / "summary_stats.csv", index=False)
    paired = _paired(df)
    paired.to_csv(out_dir / "paired_delta_vs_official_uniform.csv", index=False)

    scene_stats = (
        df.groupby(["method", "scene_id"], as_index=False)
        .agg(ME_percent_mean=("ME_percent", "mean"), ME_percent_std=("ME_percent", "std"), n=("ME_percent", "count"))
    )
    scene_stats.to_csv(out_dir / "per_scene_summary.csv", index=False)

    summary = {
        "status": "ok",
        "official_dir": str(official_dir),
        "payload_count": len(payload_paths),
        "rows": int(len(df)),
        "methods": stats.to_dict(orient="records"),
        "paired_vs_official_uniform": paired.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
