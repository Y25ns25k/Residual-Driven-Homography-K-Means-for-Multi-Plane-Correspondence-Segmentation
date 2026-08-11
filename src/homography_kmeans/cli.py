from __future__ import annotations

import argparse

from .experiment import (
    run_adelaide_experiment,
    run_conservative_calibration,
    run_merge_calibration,
    run_synthetic_experiment,
    run_threshold_sweep,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Residual-driven Homography K-Means experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="Run synthetic experiment")
    syn.add_argument("--config", default="configs/synthetic_quick.yml")
    syn.add_argument("--run-name", default=None)

    sweep = sub.add_parser("sweep", help="Run threshold sweep")
    sweep.add_argument("--config", default="configs/synthetic_full.yml")
    sweep.add_argument("--run-name", default=None)

    calib = sub.add_parser("merge-calibration", help="Run merge calibration diagnostics")
    calib.add_argument("--config", default="configs/synthetic_full.yml")
    calib.add_argument("--run-name", default=None)

    conservative = sub.add_parser("conservative-calibration", help="Run conservative HKM calibration diagnostics")
    conservative.add_argument("--config", default="configs/synthetic_full.yml")
    conservative.add_argument("--run-name", default=None)

    adel = sub.add_parser("adelaide", help="Run AdelaideRMF validation if local data exists")
    adel.add_argument("--config", default="configs/adelaide.yml")
    adel.add_argument("--data", default="data/adelaide")
    adel.add_argument("--run-name", default=None)

    args = parser.parse_args(argv)
    if args.command == "synthetic":
        run_dir = run_synthetic_experiment(args.config, args.run_name)
    elif args.command == "sweep":
        run_dir = run_threshold_sweep(args.config, args.run_name)
    elif args.command == "adelaide":
        run_dir = run_adelaide_experiment(args.config, args.data, args.run_name)
    elif args.command == "merge-calibration":
        run_dir = run_merge_calibration(args.config, args.run_name)
    else:
        run_dir = run_conservative_calibration(args.config, args.run_name)
    print(f"outputs written to {run_dir}")


if __name__ == "__main__":
    main()
