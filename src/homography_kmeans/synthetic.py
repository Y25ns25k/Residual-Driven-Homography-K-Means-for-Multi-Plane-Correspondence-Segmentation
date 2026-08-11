from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import apply_homography, normalize_homography, plane_induced_homography


@dataclass(frozen=True)
class SyntheticScene:
    scene_id: str
    difficulty: str
    x1: np.ndarray
    x2: np.ndarray
    gt_labels: np.ndarray
    gt_homographies: list[np.ndarray]
    image_shape: tuple[int, int]


def _intrinsics(image_shape: tuple[int, int], focal: float | None = None) -> np.ndarray:
    h, w = image_shape
    f = float(focal if focal is not None else 0.95 * max(h, w))
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _random_homography(rng: np.random.Generator, difficulty: str) -> np.ndarray:
    scale = {"easy": 0.00015, "medium": 0.00025, "hard": 0.00035}[difficulty]
    angle = rng.normal(0.0, {"easy": 0.05, "medium": 0.08, "hard": 0.11}[difficulty])
    c, s = np.cos(angle), np.sin(angle)
    A = np.array([[c, -s], [s, c]]) @ np.diag(rng.uniform(0.92, 1.08, 2))
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = rng.uniform(-45, 45, 2)
    H[2, :2] = rng.uniform(-scale, scale, 2)
    return normalize_homography(H)


def make_plane_homographies(
    difficulty: str,
    image_shape: tuple[int, int],
    rng: np.random.Generator,
    generator: str = "physical",
) -> list[np.ndarray]:
    if difficulty == "easy":
        K = 2
    elif difficulty == "medium":
        K = 3
    elif difficulty == "hard":
        K = 3
    else:
        raise ValueError("difficulty must be easy, medium, or hard")

    if generator == "random":
        return [_random_homography(rng, difficulty) for _ in range(K)]
    if generator != "physical":
        raise ValueError("generator must be physical or random")

    Kcam = _intrinsics(image_shape)
    motion_scale = {"easy": 1.0, "medium": 1.25, "hard": 1.55}[difficulty]
    R = _rotation_matrix(0.025 * motion_scale, -0.035 * motion_scale, 0.012 * motion_scale)
    t = np.array([0.11, -0.035, 0.22], dtype=np.float64) * motion_scale
    normals = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.28, 0.02, 1.0]),
        np.array([-0.22, 0.18, 1.0]),
    ]
    if difficulty == "hard":
        normals = [normals[0], np.array([0.12, 0.05, 1.0]), np.array([-0.12, 0.04, 1.0])]
    distances = [-3.0, -3.7, -4.5]
    Hs: list[np.ndarray] = []
    for k in range(K):
        normal = normals[k] / np.linalg.norm(normals[k])
        Hs.append(plane_induced_homography(Kcam, R, t, normal, distances[k]))
    return Hs


def _regions_for_k(K: int, image_shape: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    h, w = image_shape
    if K == 1:
        return [(0, w, 0, h)]
    if K == 2:
        return [(20, w * 0.52, 20, h - 20), (w * 0.42, w - 20, 20, h - 20)]
    return [
        (20, w * 0.48, 20, h * 0.62),
        (w * 0.36, w - 20, 20, h * 0.68),
        (w * 0.18, w - 30, h * 0.52, h - 20),
    ]


def generate_synthetic_scene(
    scene_id: str,
    difficulty: str,
    image_shape: tuple[int, int] = (480, 640),
    points_per_plane: int = 60,
    noise_std: float = 0.5,
    outlier_ratio: float = 0.15,
    seed: int = 42,
    generator: str = "physical",
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    Hs = make_plane_homographies(difficulty, image_shape, rng, generator=generator)
    h, w = image_shape
    regions = _regions_for_k(len(Hs), image_shape)
    all_x1: list[np.ndarray] = []
    all_x2: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for k, H in enumerate(Hs):
        need = int(points_per_plane)
        pts1: list[np.ndarray] = []
        pts2: list[np.ndarray] = []
        xmin, xmax, ymin, ymax = regions[k]
        attempts = 0
        while sum(len(p) for p in pts1) < need and attempts < 200:
            attempts += 1
            batch = max(need * 3, 64)
            src = np.column_stack([rng.uniform(xmin, xmax, batch), rng.uniform(ymin, ymax, batch)])
            dst = apply_homography(H, src)
            valid = (
                np.isfinite(dst).all(axis=1)
                & (dst[:, 0] >= 2)
                & (dst[:, 0] < w - 2)
                & (dst[:, 1] >= 2)
                & (dst[:, 1] < h - 2)
            )
            src = src[valid]
            dst = dst[valid]
            take = min(len(src), need - sum(len(p) for p in pts1))
            if take > 0:
                pts1.append(src[:take])
                pts2.append(dst[:take])
        if pts1:
            src_k = np.vstack(pts1)
            dst_k = np.vstack(pts2) + rng.normal(0.0, noise_std, (len(src_k), 2))
            all_x1.append(src_k)
            all_x2.append(dst_k)
            labels.append(np.full(len(src_k), k, dtype=np.int32))

    if not all_x1:
        raise RuntimeError("synthetic scene generation produced no valid correspondences")

    x1 = np.vstack(all_x1)
    x2 = np.vstack(all_x2)
    gt_labels = np.concatenate(labels)
    n_out = int(round(outlier_ratio * len(x1) / max(1e-12, 1.0 - outlier_ratio)))
    if n_out > 0:
        out1 = np.column_stack([rng.uniform(0, w, n_out), rng.uniform(0, h, n_out)])
        out2 = np.column_stack([rng.uniform(0, w, n_out), rng.uniform(0, h, n_out)])
        x1 = np.vstack([x1, out1])
        x2 = np.vstack([x2, out2])
        gt_labels = np.concatenate([gt_labels, np.full(n_out, -1, dtype=np.int32)])

    order = rng.permutation(len(x1))
    return SyntheticScene(scene_id, difficulty, x1[order], x2[order], gt_labels[order], Hs, image_shape)


def generate_synthetic_suite(config: dict, seed: int | None = None) -> list[SyntheticScene]:
    synth = config.get("synthetic", {})
    image_shape = tuple(config.get("image_shape", [480, 640]))
    base_seed = int(config.get("seed", 42) if seed is None else seed)
    scenes: list[SyntheticScene] = []
    scene_idx = 0
    for difficulty in synth.get("difficulties", ["easy", "medium", "hard"]):
        count = int(synth.get("scenes_per_difficulty", 1))
        for local in range(count):
            ppp = synth.get("points_per_plane", {}).get(difficulty, synth.get("points_per_plane", 60))
            noise = synth.get("noise_std", {}).get(difficulty, synth.get("noise_std", 0.5))
            outliers = synth.get("outlier_ratio", {}).get(difficulty, synth.get("outlier_ratio", 0.15))
            scenes.append(
                generate_synthetic_scene(
                    f"{difficulty}_{local:03d}",
                    difficulty,
                    image_shape=image_shape,
                    points_per_plane=int(ppp),
                    noise_std=float(noise),
                    outlier_ratio=float(outliers),
                    seed=base_seed + scene_idx * 97,
                    generator=str(synth.get("generator", "physical")),
                )
            )
            scene_idx += 1
    return scenes
