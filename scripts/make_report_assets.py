from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.experiment import write_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    write_reports(summary, run_dir)
    print("report/project_report.md and report/final_story.md updated")


if __name__ == "__main__":
    main()
