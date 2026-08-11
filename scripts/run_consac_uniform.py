from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONSAC_DIR = ROOT / "third_party" / "consac"
SCENES = [
    "barrsmith",
    "bonhall",
    "bonython",
    "elderhalla",
    "elderhallb",
    "hartley",
    "johnsona",
    "johnsonb",
    "ladysymon",
    "library",
    "napiera",
    "napierb",
    "neem",
    "nese",
    "oldclassicswing",
    "physics",
    "sene",
    "unihouse",
    "unionhouse",
]


def _git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(CONSAC_DIR), "rev-parse", "HEAD"], text=True).strip()


def _parse_stdout(stdout: str, runcount: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    miss_values = [float(v) for v in re.findall(r"miss\. rate:\s*([0-9.]+)", stdout)]
    elapsed_values = [float(v) for v in re.findall(r"time elapsed:\s*([0-9.]+)\s*seconds", stdout)]
    rows = []
    expected = len(SCENES) * int(runcount)
    for idx, miss in enumerate(miss_values):
        scene_index = idx // int(runcount)
        run_index = idx % int(runcount)
        scene = SCENES[scene_index] if scene_index < len(SCENES) else f"unknown_{scene_index}"
        rows.append(
            {
                "scene": scene,
                "scene_index": int(scene_index),
                "run_index": int(run_index),
                "ME_percent": float(miss),
                "runtime": float(elapsed_values[idx]) if idx < len(elapsed_values) else np.nan,
            }
        )
    per_scene = pd.DataFrame(rows)
    if per_scene.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            per_scene.groupby("scene", as_index=False)
            .agg(
                ME_percent_mean=("ME_percent", "mean"),
                ME_percent_std=("ME_percent", "std"),
                runtime_mean=("runtime", "mean"),
                n=("ME_percent", "count"),
            )
            .sort_values("scene")
        )
    overall = {
        "status": "ok" if len(miss_values) == expected else "partial",
        "expected_values": expected,
        "parsed_values": len(miss_values),
        "ME_percent_mean_all_runs": float(np.mean(miss_values)) if miss_values else float("nan"),
        "ME_percent_std_all_runs": float(np.std(miss_values, ddof=1)) if len(miss_values) > 1 else 0.0,
        "ME_percent_mean_scene_means": float(summary["ME_percent_mean"].mean()) if not summary.empty else float("nan"),
        "ME_percent_std_scene_means": float(summary["ME_percent_mean"].std(ddof=1)) if len(summary) > 1 else 0.0,
        "runtime_total_reported": float(sum(elapsed_values)) if elapsed_values else float("nan"),
        "n_scenes": int(len(summary)),
        "runcount": int(runcount),
    }
    return per_scene, summary, overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--runcount", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    run_name = args.out or f"outputs/official_consac_uniform_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = (ROOT / run_name).resolve() if not Path(run_name).is_absolute() else Path(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(exist_ok=True)
    for stale in payload_dir.glob("*.npz"):
        stale.unlink()

    commit = _git_commit()
    (out_dir / "consac_commit.txt").write_text(commit + "\n", encoding="utf-8")

    cmd = [
        args.python,
        "evaluate_homography.py",
        "--dataset_path",
        str(Path(args.data)),
        "--runcount",
        str(int(args.runcount)),
        "--uniform",
        "--resultdir",
        str(payload_dir),
    ]
    if args.cpu:
        cmd.append("--cpu")
    cmd.extend(args.extra or [])

    (out_dir / "command.txt").write_text(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, cwd=CONSAC_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = proc.stdout
    (out_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (out_dir / "returncode.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")

    per_scene, scene_summary, overall = _parse_stdout(stdout, args.runcount)
    per_scene.to_csv(out_dir / "official_uniform_per_scene.csv", index=False)
    scene_summary.to_csv(out_dir / "official_uniform_scene_summary.csv", index=False)
    overall["returncode"] = int(proc.returncode)
    overall["commit"] = commit
    overall["payload_count"] = len(list(payload_dir.glob("*.npz")))
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    summary_rows = [
        {
            "method": "official_consac_uniform",
            "ME_percent_mean": overall["ME_percent_mean_scene_means"],
            "ME_percent_std": overall["ME_percent_std_scene_means"],
            "n_scenes": overall["n_scenes"],
            "runcount": overall["runcount"],
            "payload_count": overall["payload_count"],
            "returncode": int(proc.returncode),
        }
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "official_uniform_summary.csv", index=False)
    print(json.dumps(overall, indent=2))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
