from __future__ import annotations

from pathlib import Path

import yaml
import numpy as np

from src.homography_kmeans.experiment import run_synthetic_experiment
from src.homography_kmeans.hkm import _assign_conservative
from src.homography_kmeans.hkm import ResidualHomographyKMeans
from src.homography_kmeans.synthetic import generate_synthetic_scene


def test_hkm_deterministic_labels_fixed_seed():
    scene = generate_synthetic_scene("det", "easy", points_per_plane=35, noise_std=0.25, outlier_ratio=0.05, seed=14)
    config = {
        "ransac": {"threshold": 3.0, "max_iterations": 700, "min_support": 18, "max_models": 4},
        "hkm": {"max_iterations": 8, "min_support": 18, "tau_abs": 4.0, "tau_norm": 3.0},
    }
    a = ResidualHomographyKMeans(config, random_state=11).fit(scene.x1, scene.x2, image_shape=scene.image_shape)
    b = ResidualHomographyKMeans(config, random_state=11).fit(scene.x1, scene.x2, image_shape=scene.image_shape)
    assert np.array_equal(a.labels, b.labels)
    assert len(a.history) >= 1


def test_cli_quick_synthetic_outputs_metrics_and_summary(tmp_path: Path):
    cfg = {
        "run_name": "tiny",
        "output_root": str(tmp_path),
        "seed": 5,
        "image_shape": [180, 240],
        "synthetic": {
            "generator": "physical",
            "difficulties": ["easy"],
            "scenes_per_difficulty": 1,
            "points_per_plane": {"easy": 24},
            "noise_std": {"easy": 0.2},
            "outlier_ratio": {"easy": 0.05},
        },
        "methods": ["sequential_ransac", "residual_hkm_energy_merge"],
        "write_reports": False,
        "ransac": {"threshold": 3.0, "max_iterations": 350, "min_support": 12, "max_models": 3},
        "hkm": {"max_iterations": 5, "min_support": 12, "tau_abs": 4.0, "tau_norm": 3.0},
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_dir = run_synthetic_experiment(cfg_path)
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "figures" / "residual_histograms.png").exists()


def test_conservative_assignment_requires_margin_to_switch():
    errors = np.array(
        [
            [10.0, 9.0],
            [10.0, 7.0],
            [8.0, 3.0],
        ],
        dtype=float,
    )
    scales = np.ones(2)
    previous = np.array([0, 0, -1], dtype=np.int32)
    labels = _assign_conservative(
        errors,
        scales,
        previous,
        tau_abs=20.0,
        tau_norm=20.0,
        reassignment_margin=0.15,
        scale_adaptive=True,
    )
    assert labels.tolist() == [0, 1, 1]
