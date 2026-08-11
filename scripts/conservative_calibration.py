from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.experiment import run_conservative_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_full.yml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_dir = run_conservative_calibration(args.config, args.run_name)
    print(f"outputs written to {run_dir}")


if __name__ == "__main__":
    main()
