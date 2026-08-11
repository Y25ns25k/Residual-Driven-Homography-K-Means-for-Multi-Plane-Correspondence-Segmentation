from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["ME", "SegAcc", "CountAcc", "AbsK", "OverSeg", "UnderSeg", "Runtime"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "metrics.csv")
    rows = []
    for method, sub in df.groupby("method"):
        row = {"method": method, "n": len(sub)}
        for metric in METRICS:
            if metric in sub:
                row[f"{metric}_mean"] = sub[metric].mean()
                row[f"{metric}_std"] = sub[metric].std(ddof=1) if len(sub) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    out = run_dir / "summary_table.csv"
    summary.to_csv(out, index=False)
    print(f"table written to {out}")


if __name__ == "__main__":
    main()
