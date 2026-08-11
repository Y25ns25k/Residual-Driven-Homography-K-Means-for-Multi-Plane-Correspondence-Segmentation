from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


METHOD_ORDER = ["global_ransac", "sequential_ransac", "residual_hkm_v2"]
METHOD_TITLES = {
    "global_ransac": "Global RANSAC\n(single H)",
    "single_homography_ransac": "Global RANSAC\n(single H)",
    "sequential_ransac": "Sequential\nRANSAC",
    "residual_hkm_v2": "Residual\nHKM v2",
}
METHOD_COLORS = {
    "global_ransac": "#8E8E93",
    "sequential_ransac": "#F58518",
    "residual_hkm_v2": "#2E7D32",
}
QUAL_SCENES = ["bonhall", "bonython", "barrsmith"]


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _fmt_pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        vals: list[str] = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        sub = metrics[metrics["method"] == method]
        if sub.empty:
            continue
        row: dict[str, Any] = {"method": method, "display_method": METHOD_TITLES.get(method, method).replace("\n", " ")}
        for metric in ["ME_percent", "SegAcc", "CountAcc", "AbsK", "OverSeg", "UnderSeg", "Runtime"]:
            vals = pd.to_numeric(sub[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.count() > 1 else 0.0
        row["n"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def _display_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Method": row["display_method"],
                "ME_percent mean +/- std": _fmt_pm(row["ME_percent_mean"], row["ME_percent_std"]),
                "SegAcc mean": row["SegAcc_mean"],
                "CountAcc mean": row["CountAcc_mean"],
                "AbsK mean": row["AbsK_mean"],
                "OverSeg mean": row["OverSeg_mean"],
                "UnderSeg mean": row["UnderSeg_mean"],
                "Runtime mean": row["Runtime_mean"],
            }
        )
    return pd.DataFrame(rows)


def _paired_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("global_ransac", "sequential_ransac"),
        ("sequential_ransac", "residual_hkm_v2"),
        ("global_ransac", "residual_hkm_v2"),
    ]
    rows: list[dict[str, Any]] = []
    eps = 1e-6
    for baseline, method in pairs:
        base = metrics[metrics["method"] == baseline][["scene_id", "seed", "ME_percent"]].rename(
            columns={"ME_percent": "ME_baseline"}
        )
        cur = metrics[metrics["method"] == method][["scene_id", "seed", "ME_percent"]].rename(
            columns={"ME_percent": "ME_method"}
        )
        merged = cur.merge(base, on=["scene_id", "seed"], how="inner")
        delta = (merged["ME_method"] - merged["ME_baseline"]).to_numpy(dtype=np.float64)
        stat = np.nan
        pvalue = np.nan
        nonzero = delta[np.abs(delta) > eps]
        if wilcoxon is not None and len(nonzero):
            res = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
            stat = float(res.statistic)
            pvalue = float(res.pvalue)
        rows.append(
            {
                "comparison": f"{method} - {baseline}",
                "baseline": baseline,
                "method": method,
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


def _plot_me(summary: pd.DataFrame, path: Path) -> None:
    methods = summary["method"].tolist()
    means = summary["ME_percent_mean"].to_numpy(dtype=float)
    stds = summary["ME_percent_std"].to_numpy(dtype=float)
    colors = [METHOD_COLORS.get(m, "#4C78A8") for m in methods]
    labels = [METHOD_TITLES.get(m, m) for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#333333", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ME_percent (lower is better)")
    ax.set_title("AdelaideRMF-H: from single-H RANSAC to residual HKM")
    ax.grid(axis="y", alpha=0.25)
    for xi, val in zip(x, means):
        ax.text(xi, val + max(stds.max() * 0.05, 0.5), f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_all_metrics(summary: pd.DataFrame, path: Path) -> None:
    plot_specs = [
        ("ME_percent_mean", "ME_percent", "lower better"),
        ("SegAcc_mean", "SegAcc", "higher better"),
        ("CountAcc_mean", "CountAcc", "higher better"),
        ("AbsK_mean", "AbsK", "lower better"),
    ]
    methods = summary["method"].tolist()
    labels = [METHOD_TITLES.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "#4C78A8") for m in methods]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), constrained_layout=True)
    for ax, (col, title, subtitle) in zip(axes, plot_specs):
        vals = summary[col].to_numpy(dtype=float)
        x = np.arange(len(methods))
        ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        for xi, val in zip(x, vals):
            ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("AdelaideRMF-H final baseline comparison", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _label_values(labels: np.ndarray) -> list[int]:
    return [int(v) for v in sorted(np.unique(labels)) if int(v) >= 0]


def _palette(labels: np.ndarray) -> dict[int, Any]:
    cmap = plt.get_cmap("tab20")
    colors = {-1: (0.35, 0.35, 0.35, 1.0)}
    for i, label in enumerate(_label_values(labels)):
        colors[int(label)] = cmap(i % 20)
    return colors


def _map_pred_labels(pred: np.ndarray, mapping_pairs: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    mapped = np.full(len(pred), -1, dtype=np.int32)
    pairs = np.asarray(mapping_pairs).reshape(-1, 2) if np.asarray(mapping_pairs).size else np.empty((0, 2), dtype=np.int32)
    mapping = {int(p): int(g) for p, g in pairs}
    next_label = max(_label_values(gt_labels) or [0]) + 1
    extras: dict[int, int] = {}
    for label in sorted(np.unique(pred)):
        lab = int(label)
        if lab < 0:
            mapped[pred == lab] = -1
        elif lab in mapping:
            mapped[pred == lab] = mapping[lab]
        else:
            if lab not in extras:
                extras[lab] = next_label
                next_label += 1
            mapped[pred == lab] = extras[lab]
    return mapped


def _colors_for(labels: np.ndarray, colors: dict[int, Any]) -> list[Any]:
    return [colors.get(int(v), (0.1, 0.1, 0.1, 1.0)) for v in labels]


def _npz_path(results: Path, scene: str, seed: int, method: str) -> Path:
    return results / "labels" / f"{scene}_seed{int(seed)}_{method}.npz"


def _choose_seed(metrics: pd.DataFrame, scene: str) -> int:
    sub = metrics[(metrics["scene_id"].astype(str).str.lower() == scene.lower())]
    pair = sub[sub["method"].isin(["sequential_ransac", "residual_hkm_v2"])]
    pivot = pair.pivot_table(index="seed", columns="method", values="ME_percent", aggfunc="first").dropna()
    if pivot.empty:
        return int(sub["seed"].iloc[0])
    pivot["delta"] = pivot["residual_hkm_v2"] - pivot["sequential_ransac"]
    mean_delta = float(pivot["delta"].mean())
    return int((pivot["delta"] - mean_delta).abs().sort_values().index[0])


def _plot_panel(ax: plt.Axes, x1: np.ndarray, labels: np.ndarray, colors: dict[int, Any], title: str) -> None:
    ax.scatter(x1[:, 0], x1[:, 1], s=8, c=_colors_for(labels, colors), linewidths=0.0, alpha=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def _make_qualitative_grid(metrics: pd.DataFrame, results: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(len(QUAL_SCENES), 4, figsize=(14, 9), constrained_layout=True)
    for r, scene in enumerate(QUAL_SCENES):
        seed = _choose_seed(metrics, scene)
        payloads = {method: dict(np.load(_npz_path(results, scene, seed, method), allow_pickle=False)) for method in METHOD_ORDER}
        gt = np.asarray(payloads["sequential_ransac"]["gt_labels"], dtype=np.int32)
        x1 = np.asarray(payloads["sequential_ransac"]["x1"], dtype=np.float64)
        colors = _palette(gt)
        _plot_panel(axes[r, 0], x1, gt, colors, f"{scene}\nGT, K={len(_label_values(gt))}")
        for c, method in enumerate(METHOD_ORDER, start=1):
            item = payloads[method]
            pred = np.asarray(item["pred_labels"], dtype=np.int32)
            disp = _map_pred_labels(pred, item["label_mapping"], gt)
            me = 100.0 * float(np.asarray(item["ME"]).item())
            k_est = int(round(float(np.asarray(item["K_est"]).item())))
            _plot_panel(axes[r, c], x1, disp, colors, f"{METHOD_TITLES[method].replace(chr(10), ' ')}\nME={me:.1f}%, K={k_est}")
    fig.suptitle("Representative AdelaideRMF-H labels: GT vs Global RANSAC vs Sequential RANSAC vs Residual HKM v2", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="presentation_assets/final_baseline_comparison")
    args = parser.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(results / "per_scene_results.csv")
    metrics = metrics[_bool_series(metrics["include_outliers"])].copy()
    metrics = metrics[metrics["method"].isin(METHOD_ORDER)].copy()
    summary = _main_summary(metrics)
    summary.to_csv(out / "main_baseline_table.csv", index=False)
    display = _display_table(summary)
    _write_markdown(display, out / "main_baseline_table.md")

    paired = _paired_delta(metrics)
    paired.to_csv(out / "paired_delta_summary.csv", index=False)
    paired_display = paired[
        [
            "comparison",
            "delta_ME_percent_mean",
            "delta_ME_percent_std",
            "win_count",
            "loss_count",
            "tie_count",
            "wilcoxon_pvalue",
            "n",
        ]
    ].copy()
    _write_markdown(paired_display, out / "paired_delta_summary.md")

    _plot_me(summary, out / "main_baseline_bar_me.png")
    _plot_all_metrics(summary, out / "main_baseline_bar_all_metrics.png")
    _make_qualitative_grid(metrics, results, out / "global_seq_hkm_qualitative_grid.png")

    readme = [
        "# Final Baseline Comparison",
        "",
        f"Source results: `{results.as_posix()}`",
        "",
        "Main comparison uses AdelaideRMF-H, 19 scenes, 5 seeds, `include_outliers=true`.",
        "",
        "Methods:",
        "",
        "- `global_ransac`: Single-Homography RANSAC fitted once to all correspondences.",
        "- `sequential_ransac`: greedy repeated homography RANSAC.",
        "- `residual_hkm_v2`: residual-driven HKM with rank-4 candidate gate and ICM Potts smoothing.",
        "",
        "This is a controlled baseline progression, not a SOTA comparison.",
        "",
        "Files:",
        "",
        "- `main_baseline_table.md` / `main_baseline_table.csv`",
        "- `paired_delta_summary.md` / `paired_delta_summary.csv`",
        "- `main_baseline_bar_me.png`",
        "- `main_baseline_bar_all_metrics.png`",
        "- `global_seq_hkm_qualitative_grid.png`",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    summary_json = {
        "results": str(results),
        "summary": summary.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
