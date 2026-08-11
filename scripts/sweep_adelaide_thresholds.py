from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homography_kmeans.experiment import run_adelaide_threshold_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/adelaidermf")
    parser.add_argument("--config", default="configs/adelaide.yml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_name = args.run_name or f"adelaide_threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = run_adelaide_threshold_sweep(args.config, args.data, run_name)
    print(f"outputs written to {run_dir}")


if __name__ == "__main__":
    main()
