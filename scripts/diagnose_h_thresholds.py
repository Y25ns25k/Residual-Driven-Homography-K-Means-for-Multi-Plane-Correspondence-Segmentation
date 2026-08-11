from __future__ import annotations

import argparse
import copy
import json
import sys
import time
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

from homography_kmeans.adelaide import filter_adelaide_scenes, load_adelaide_directory_report
from homography_kmeans.auto_threshold import estimate_scene_sigma, scale_config_thresholds
from homography_kmeans.experiment import (
    _adelaide_method_seed,
    _adelaide_seed_values,
    _method_fit,
    _seed_method_name,
    load_config,
)
from homography_kmeans.metrics import evaluate_segmentation


HIGH_NOISE_SCENES = {"physics", "elderhalla", "elderhallb", "napierb", "barrsmith", "neem"}


FIXED_SETTINGS = {
    "base": {"ransac_threshold": 2.5, "tau_abs": 4.0, "tau_norm": 3.0, "sigma_min": 0.75},
    "mid": {"ransac_threshold": 3.5, "tau_abs": 5.5, "tau_norm": 3.5, "sigma_min": 0.75},
    "loose": {"ransac_threshold": 5.0, "tau_abs": 7.0, "tau_norm": 4.0, "sigma_min": 0.75},
    "very_loose": {"ransac_threshold": 7.0, "tau_abs": 9.0, "tau_norm": 4.5, "sigma_min": 0.75},
}


def _fixed_config(config: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("ransac", {})["threshold"] = float(values["ransac_threshold"])
    hkm = cfg.setdefault("hkm", {})
    hkm["tau_abs"] = float(values["tau_abs"])
    hkm["tau_norm"] = float(values["tau_norm"])
    hkm["sigma_min"] = float(values["sigma_min"])
    return cfg


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in ["ME_percent", "SegAcc", "CountAcc", "AbsK", "OverSeg", "UnderSeg", "Runtime"]:
            vals = pd.to_numeric(sub[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.count() > 1 else 0.0
        row["n"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_vs_base(df: pd.DataFrame, method: str) -> pd.DataFrame:
    base = df[(df["method"] == method) & (df["setting"] == "base")][["scene_id", "seed", "ME_percent"]].rename(
        columns={"ME_percent": "ME_base"}
    )
    rows: list[dict[str, Any]] = []
    eps = 1e-6
    for setting, sub in df[df["method"] == method].groupby("setting"):
        merged = sub[["scene_id", "seed", "ME_percent"]].merge(base, on=["scene_id", "seed"], how="inner")
        if merged.empty:
            continue
        delta = (merged["ME_percent"] - merged["ME_base"]).to_numpy(dtype=np.float64)
        stat = np.nan
        pvalue = np.nan
        nonzero = delta[np.abs(delta) > eps]
        if wilcoxon is not None and len(nonzero):
            res = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
            stat = float(res.statistic)
            pvalue = float(res.pvalue)
        rows.append(
            {
                "method": method,
                "setting": setting,
                "delta_ME_percent_mean": float(np.mean(delta)),
                "delta_ME_percent_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "win_count": int(np.sum(delta < -eps)),
                "loss_count": int(np.sum(delta > eps)),
                "tie_count": int(np.sum(np.abs(delta) <= eps)),
                "wilcoxon_stat": stat,
                "wilcoxon_pvalue": pvalue,
                "n": int(len(delta)),
            }
        )
    return pd.DataFrame(rows)


def _scene_delta_table(df: pd.DataFrame, method: str) -> pd.DataFrame:
    scene = _summary(df[df["method"] == method], ["setting", "scene_id", "is_high_noise"])
    base = scene[scene["setting"] == "base"][["scene_id", "ME_percent_mean"]].rename(
        columns={"ME_percent_mean": "ME_base_mean"}
    )
    out = scene.merge(base, on="scene_id", how="left")
    out["delta_vs_base_pp"] = out["ME_percent_mean"] - out["ME_base_mean"]
    return out.sort_values(["setting", "delta_vs_base_pp", "scene_id"]).reset_index(drop=True)


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/adelaide.yml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--methods", nargs="+", default=["residual_hkm_v2"])
    parser.add_argument("--settings", nargs="+", default=["base", "mid", "loose", "very_loose", "adaptive"])
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.out or f"outputs/h_threshold_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.yml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    load_report = load_adelaide_directory_report(args.data)
    scenes, missing = filter_adelaide_scenes(load_report.scenes, "homography")
    if missing:
        print(f"[threshold] missing H scenes: {missing}")
    seeds = _adelaide_seed_values(config)

    rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for setting in args.settings:
        print(f"[threshold] setting={setting}", flush=True)
        for seed_index, seed in enumerate(seeds):
            for scene in scenes:
                if setting == "adaptive":
                    pilot_seed = _adelaide_method_seed(seed, scene.scene_id, "adaptive_threshold_pilot")
                    scale = estimate_scene_sigma(
                        scene.x1,
                        scene.x2,
                        random_state=pilot_seed,
                        pilot_threshold=5.0,
                        max_iterations=int(config.get("ransac", {}).get("max_iterations", 2500)),
                        min_support=max(8, int(config.get("ransac", {}).get("min_support", 20)) // 2),
                        clip_range=(0.6, 2.5),
                    )
                    cfg = scale_config_thresholds(config, scale.sigma_hat)
                    scale_rows.append(
                        {
                            "setting": setting,
                            "scene_id": scene.scene_id,
                            "seed": int(seed),
                            "sigma_hat": scale.sigma_hat,
                            "sigma_raw": scale.sigma_raw,
                            "pilot_inliers": scale.pilot_inliers,
                            "pilot_success": scale.pilot_success,
                            "is_high_noise": scene.scene_id.lower() in HIGH_NOISE_SCENES,
                        }
                    )
                else:
                    cfg = _fixed_config(config, FIXED_SETTINGS[setting])
                for method in args.methods:
                    method_seed = _adelaide_method_seed(seed, scene.scene_id, _seed_method_name(method))
                    Hs, labels, residuals, runtime, diagnostics = _method_fit(scene, method, cfg, method_seed)
                    metrics = evaluate_segmentation(
                        scene.labels,
                        labels,
                        pred_homographies=Hs,
                        x1=scene.x1,
                        x2=scene.x2,
                        image_shape=scene.image_shape or tuple(cfg.get("image_shape", [480, 640])),
                        include_outliers=True,
                        runtime=runtime,
                    )
                    metrics["ME_percent"] = 100.0 * metrics["ME"]
                    rows.append(
                        {
                            "setting": setting,
                            "scene_id": scene.scene_id,
                            "seed": int(seed),
                            "seed_index": int(seed_index),
                            "method": method,
                            "is_high_noise": scene.scene_id.lower() in HIGH_NOISE_SCENES,
                            **metrics,
                            "diagnostics_json": json.dumps(diagnostics, default=float),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    scales = pd.DataFrame(scale_rows)
    scales.to_csv(run_dir / "adaptive_sigma.csv", index=False)

    overall = _summary(df, ["setting", "method"])
    high_noise = _summary(df[df["is_high_noise"]], ["setting", "method"])
    normal = _summary(df[~df["is_high_noise"]], ["setting", "method"])
    scene_delta_frames = [_scene_delta_table(df, method) for method in args.methods]
    scene_delta = pd.concat(scene_delta_frames, ignore_index=True) if scene_delta_frames else pd.DataFrame()
    paired = pd.concat([_paired_vs_base(df, method) for method in args.methods], ignore_index=True)

    overall.to_csv(run_dir / "summary_overall.csv", index=False)
    high_noise.to_csv(run_dir / "summary_high_noise_scenes.csv", index=False)
    normal.to_csv(run_dir / "summary_other_scenes.csv", index=False)
    scene_delta.to_csv(run_dir / "scene_delta_vs_base.csv", index=False)
    paired.to_csv(run_dir / "paired_vs_base.csv", index=False)

    _write_markdown(overall, run_dir / "summary_overall.md")
    _write_markdown(high_noise, run_dir / "summary_high_noise_scenes.md")
    _write_markdown(normal, run_dir / "summary_other_scenes.md")
    _write_markdown(paired, run_dir / "paired_vs_base.md")

    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "elapsed_sec": time.perf_counter() - t_all,
        "methods": args.methods,
        "settings": args.settings,
        "n_rows": int(len(df)),
        "overall": overall.to_dict(orient="records"),
        "high_noise": high_noise.to_dict(orient="records"),
        "paired_vs_base": paired.to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
