from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


METHOD_ORDER = [
    "sequential_ransac",
    "residual_hkm_no_merge",
    "residual_hkm_conservative",
    "residual_hkm_functional_merge",
    "residual_hkm_energy_merge",
]

METHOD_LABELS = {
    "sequential_ransac": "Sequential RANSAC",
    "residual_hkm_no_merge": "HKM no merge",
    "residual_hkm_conservative": "HKM conservative",
    "residual_hkm_functional_merge": "HKM functional merge",
    "residual_hkm_energy_merge": "HKM energy merge",
}

SUBSET_LABELS = {
    "homography": "H-only",
    "fundamental": "F-only",
    "all": "H+F all",
}

TABLE_COLS = [
    "method",
    "ME_percent",
    "SegAcc",
    "CountAcc",
    "AbsK",
    "OverSeg",
    "UnderSeg",
    "Runtime",
]


def _read_summary(path: str | Path, subset_name: str) -> pd.DataFrame:
    base = Path(path)
    paper = base / "paper_summary.csv"
    stats = base / "summary_stats.csv"
    if paper.exists():
        df = pd.read_csv(paper)
    elif stats.exists():
        df = pd.read_csv(stats)
    else:
        raise FileNotFoundError(f"Could not find paper_summary.csv or summary_stats.csv in {base}")
    df = df.copy()
    df["source_dir"] = str(base)
    if "subset" not in df.columns:
        df["subset"] = subset_name
    return df


def _method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def _main_rows(df: pd.DataFrame, include_outliers: bool = True) -> pd.DataFrame:
    out = df[df["include_outliers"].astype(bool) == bool(include_outliers)].copy()
    out["method_order"] = out["method"].map(_method_sort_key)
    return out.sort_values(["subset", "method_order", "method"]).drop(columns=["method_order"])


def _fmt_mean_std(row: pd.Series) -> str:
    return f"{row['ME_percent_mean']:.2f} +/- {row['ME_percent_std']:.2f}"


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "method": METHOD_LABELS.get(row["method"], row["method"]),
                "ME_percent mean +/- std": _fmt_mean_std(row),
                "SegAcc mean": f"{row['SegAcc_mean']:.3f}",
                "CountAcc mean": f"{row['CountAcc_mean']:.3f}",
                "AbsK mean": f"{row['AbsK_mean']:.3f}",
                "OverSeg mean": f"{row['OverSeg_mean']:.3f}",
                "UnderSeg mean": f"{row['UnderSeg_mean']:.3f}",
                "Runtime mean": f"{row['Runtime_mean']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def _write_md_table(path: Path, title: str, df: pd.DataFrame, note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.extend(_markdown_table_lines(df))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table_lines(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _bar_grouped(df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    subsets = ["homography", "fundamental", "all"]
    x = np.arange(len(methods))
    width = 0.24
    colors = {"homography": "#4C78A8", "fundamental": "#F58518", "all": "#54A24B"}
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for i, subset in enumerate(subsets):
        sub = df[df["subset"] == subset].set_index("method")
        values = [float(sub.loc[m, metric]) if m in sub.index else np.nan for m in methods]
        ax.bar(x + (i - 1) * width, values, width, label=SUBSET_LABELS[subset], color=colors[subset])
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _h_vs_hf_figure(df: pd.DataFrame, path: Path) -> None:
    methods = ["sequential_ransac", "residual_hkm_no_merge"]
    metrics = [
        ("ME_percent_mean", "ME_percent"),
        ("SegAcc_mean", "SegAcc"),
        ("CountAcc_mean", "CountAcc"),
        ("AbsK_mean", "AbsK"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        x = np.arange(len(methods))
        width = 0.35
        for i, subset in enumerate(["homography", "all"]):
            sub = df[df["subset"] == subset].set_index("method")
            values = [float(sub.loc[m, metric]) for m in methods]
            ax.bar(x + (i - 0.5) * width, values, width, label=SUBSET_LABELS[subset])
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("H-only benchmark vs mixed H+F diagnostic", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _f_delta_figure(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    f = df[df["subset"] == "fundamental"].set_index("method")
    seq = float(f.loc["sequential_ransac", "ME_percent_mean"])
    rows = []
    for method in METHOD_ORDER:
        if method == "sequential_ransac" or method not in f.index:
            continue
        delta = float(f.loc[method, "ME_percent_mean"]) - seq
        rows.append({"method": method, "delta_ME_percent": delta})
    delta_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    colors = ["#54A24B" if v < 0 else "#E45756" for v in delta_df["delta_ME_percent"]]
    ax.bar([METHOD_LABELS.get(m, m) for m in delta_df["method"]], delta_df["delta_ME_percent"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta ME_percent vs Sequential")
    ax.set_title("F-only diagnostic: HKM variants under homography model")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return delta_df


def _best_methods_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, sub in df.groupby("subset"):
        best = sub.sort_values("ME_percent_mean").iloc[0]
        seq = sub[sub["method"] == "sequential_ransac"].iloc[0]
        rows.append(
            {
                "subset": SUBSET_LABELS.get(subset, subset),
                "best_method": METHOD_LABELS.get(best["method"], best["method"]),
                "best_ME_percent": f"{best['ME_percent_mean']:.2f} +/- {best['ME_percent_std']:.2f}",
                "sequential_ME_percent": f"{seq['ME_percent_mean']:.2f} +/- {seq['ME_percent_std']:.2f}",
                "delta_vs_sequential": f"{best['ME_percent_mean'] - seq['ME_percent_mean']:.2f}",
            }
        )
    return pd.DataFrame(rows)


def _h_vs_hf_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["sequential_ransac", "residual_hkm_no_merge"]:
        h = df[(df["subset"] == "homography") & (df["method"] == method)].iloc[0]
        all_row = df[(df["subset"] == "all") & (df["method"] == method)].iloc[0]
        rows.append(
            {
                "method": METHOD_LABELS.get(method, method),
                "H-only ME_percent": _fmt_mean_std(h),
                "H+F all ME_percent": _fmt_mean_std(all_row),
                "ME degradation H+F - H": f"{all_row['ME_percent_mean'] - h['ME_percent_mean']:.2f}",
                "H-only SegAcc": f"{h['SegAcc_mean']:.3f}",
                "H+F all SegAcc": f"{all_row['SegAcc_mean']:.3f}",
                "H-only AbsK": f"{h['AbsK_mean']:.3f}",
                "H+F all AbsK": f"{all_row['AbsK_mean']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def _write_readme(out: Path, args: argparse.Namespace, main: pd.DataFrame, f_delta: pd.DataFrame) -> None:
    h = main[main["subset"] == "homography"].set_index("method")
    f = main[main["subset"] == "fundamental"].set_index("method")
    all_df = main[main["subset"] == "all"].set_index("method")
    seq_h = h.loc["sequential_ransac"]
    no_h = h.loc["residual_hkm_no_merge"]
    seq_f = f.loc["sequential_ransac"]
    no_f = f.loc["residual_hkm_no_merge"]
    seq_all = all_df.loc["sequential_ransac"]
    no_all = all_df.loc["residual_hkm_no_merge"]
    lines = [
        "# AdelaideRMF Subset Comparison",
        "",
        f"H-only source: `{args.homography}`",
        f"F-only source: `{args.fundamental}`",
        f"H+F all source: `{args.all}`",
        "",
        "Main comparison uses `include_outliers=true` and `ME_percent = 100 * ME`.",
        "",
        "## Interpretation",
        "",
        "H-only is the main paper-comparable benchmark for this homography-fitting project.",
        "F-only is diagnostic because those scenes are fundamental-matrix data; the current model fits homographies, not epipolar geometry.",
        "H+F all is a mixed diagnostic and should not be compared directly to multi-homography papers.",
        "",
        f"On H-only, HKM no-merge improves over Sequential RANSAC: {seq_h['ME_percent_mean']:.2f}% -> {no_h['ME_percent_mean']:.2f}% ME.",
        f"On F-only, both methods degrade strongly relative to H-only: Sequential is {seq_f['ME_percent_mean']:.2f}% and HKM no-merge is {no_f['ME_percent_mean']:.2f}% ME.",
        f"On mixed H+F all, ME also rises relative to H-only: Sequential {seq_all['ME_percent_mean']:.2f}%, HKM no-merge {no_all['ME_percent_mean']:.2f}%.",
        "",
        "This pattern is consistent with model mismatch: homographies assume planar structure or pure camera rotation, while F scenes are governed by epipolar geometry.",
        "",
        "## F-only Delta vs Sequential",
        "",
        *_markdown_table_lines(f_delta.assign(method=f_delta["method"].map(lambda m: METHOD_LABELS.get(m, m)))),
        "",
        "Warning: F-only and H+F all are diagnostic stress tests, not SOTA claims and not standard homography benchmark results.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--homography", required=True)
    parser.add_argument("--fundamental", required=True)
    parser.add_argument("--all", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    h = _read_summary(args.homography, "homography")
    f = _read_summary(args.fundamental, "fundamental")
    a = _read_summary(args.all, "all")
    combined = pd.concat([h, f, a], ignore_index=True)
    main_df = _main_rows(combined, include_outliers=True)
    supplementary_df = _main_rows(combined, include_outliers=False)
    main_df.to_csv(out / "adelaide_subset_comparison_main.csv", index=False)
    supplementary_df.to_csv(out / "adelaide_subset_comparison_supplementary.csv", index=False)

    h_main = main_df[main_df["subset"] == "homography"]
    f_main = main_df[main_df["subset"] == "fundamental"]
    all_main = main_df[main_df["subset"] == "all"]
    _write_md_table(tables / "adelaide_h_main_table.md", "AdelaideRMF-H Main Benchmark", _format_table(h_main), "include_outliers=true")
    _write_md_table(tables / "adelaide_f_diagnostic_table.md", "AdelaideRMF-F Diagnostic", _format_table(f_main), "include_outliers=true; diagnostic only for a homography model.")
    _write_md_table(tables / "adelaide_all_mixed_table.md", "AdelaideRMF Mixed H+F Diagnostic", _format_table(all_main), "include_outliers=true; mixed Homography + Fundamental data.")
    hvshf = _h_vs_hf_table(main_df)
    _write_md_table(tables / "h_vs_hf_comparison_table.md", "H-only vs Mixed H+F Comparison", hvshf)
    best = _best_methods_table(main_df)
    _write_md_table(tables / "subset_best_methods_table.md", "Best Method by Subset", best)

    _bar_grouped(main_df, "ME_percent_mean", "ME_percent", "AdelaideRMF subset comparison: H main vs F/all diagnostics", figures / "subset_me_comparison.png")
    _bar_grouped(main_df, "SegAcc_mean", "SegAcc", "AdelaideRMF subset comparison: SegAcc", figures / "subset_segacc_comparison.png")
    _bar_grouped(main_df, "AbsK_mean", "AbsK", "AdelaideRMF subset comparison: K error", figures / "subset_k_error_comparison.png")
    _h_vs_hf_figure(main_df, figures / "h_vs_hf_final_model.png")
    f_delta = _f_delta_figure(main_df, figures / "f_diagnostic_delta_vs_seq.png")
    f_delta.to_csv(out / "f_diagnostic_delta_vs_seq.csv", index=False)
    _write_readme(out, args, main_df, f_delta)

    print(f"wrote tables to {tables}")
    print(f"wrote figures to {figures}")
    print(f"wrote README to {out / 'README.md'}")


if __name__ == "__main__":
    main()
