from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConsacPayload:
    scene: str
    run_index: int
    x1: np.ndarray
    x2: np.ndarray
    gt_labels: np.ndarray
    official_labels: np.ndarray
    homographies_norm: list[np.ndarray]
    homographies_pixel: list[np.ndarray]
    selected_instances: int
    official_miss_rate: float
    image_shape: tuple[int, int]
    img1_shape: tuple[int, ...]
    img2_shape: tuple[int, ...]


def normalization_matrix(image_shape: tuple[int, ...]) -> np.ndarray:
    """Return A such that x_norm_h ~ A @ x_pixel_h for CONSAC coordinates."""
    h = float(image_shape[0])
    w = float(image_shape[1])
    scale = max(h, w)
    return np.array(
        [
            [2.0 / scale, 0.0, -w / scale],
            [0.0, 2.0 / scale, -h / scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def denormalization_matrix(image_shape: tuple[int, ...]) -> np.ndarray:
    return np.linalg.inv(normalization_matrix(image_shape))


def normalized_to_pixel_points(points_norm: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    pts = np.asarray(points_norm, dtype=np.float64)
    h = float(image_shape[0])
    w = float(image_shape[1])
    scale = max(h, w)
    out = pts.copy()
    out[:, 0] = out[:, 0] * (scale / 2.0) + w / 2.0
    out[:, 1] = out[:, 1] * (scale / 2.0) + h / 2.0
    return out


def pixel_to_normalized_points(points_pixel: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    pts = np.asarray(points_pixel, dtype=np.float64)
    h = float(image_shape[0])
    w = float(image_shape[1])
    scale = max(h, w)
    out = pts.copy()
    out[:, 0] = (out[:, 0] - w / 2.0) / (scale / 2.0)
    out[:, 1] = (out[:, 1] - h / 2.0) / (scale / 2.0)
    return out


def homography_norm_to_pixel(H_norm: np.ndarray, img1_shape: tuple[int, ...], img2_shape: tuple[int, ...]) -> np.ndarray:
    """Convert CONSAC normalized-coordinate H to pixel-coordinate H.

    CONSAC uses x_norm = A @ x_pix. If H_norm maps normalized image-1
    coordinates to normalized image-2 coordinates, then:

        H_pix = A2^{-1} @ H_norm @ A1
    """
    Hn = np.asarray(H_norm, dtype=np.float64).reshape(3, 3)
    A1 = normalization_matrix(img1_shape)
    A2_inv = denormalization_matrix(img2_shape)
    Hp = A2_inv @ Hn @ A1
    if abs(Hp[2, 2]) > 1e-12:
        Hp = Hp / Hp[2, 2]
    else:
        norm = np.linalg.norm(Hp)
        if norm > 1e-12:
            Hp = Hp / norm
    return Hp


def _official_labels_to_ours(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int32).copy()
    out = np.full_like(lab, -1)
    valid = lab > 0
    out[valid] = lab[valid] - 1
    return out


def _models_from_array(models: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(models, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 9)
    Hs = []
    for row in arr:
        H = row.reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        Hs.append(H)
    return Hs


def load_consac_payload(path: str | Path) -> ConsacPayload:
    p = Path(path)
    data = np.load(p, allow_pickle=False)
    scene = str(np.asarray(data["scene_name"]).item())
    run_index = int(np.asarray(data["run_index"]).item())
    img1_shape = tuple(int(v) for v in np.asarray(data["img1_shape"]).reshape(-1))
    img2_shape = tuple(int(v) for v in np.asarray(data["img2_shape"]).reshape(-1))
    data_norm = np.asarray(data["data_norm"], dtype=np.float64)
    x1 = normalized_to_pixel_points(data_norm[:, 0:2], img1_shape)
    x2 = normalized_to_pixel_points(data_norm[:, 2:4], img2_shape)
    gt_labels = _official_labels_to_ours(np.asarray(data["labels_gt"], dtype=np.int32))
    official_labels = _official_labels_to_ours(np.asarray(data["official_estm_labels"], dtype=np.int32))
    H_norm = _models_from_array(np.asarray(data["official_models_norm"], dtype=np.float64))
    H_pix = [homography_norm_to_pixel(H, img1_shape, img2_shape) for H in H_norm]
    return ConsacPayload(
        scene=scene,
        run_index=run_index,
        x1=x1,
        x2=x2,
        gt_labels=gt_labels,
        official_labels=official_labels,
        homographies_norm=H_norm,
        homographies_pixel=H_pix,
        selected_instances=int(np.asarray(data["selected_instances"]).item()),
        official_miss_rate=float(np.asarray(data["official_miss_rate"]).item()),
        image_shape=(int(img1_shape[0]), int(img1_shape[1])),
        img1_shape=img1_shape,
        img2_shape=img2_shape,
    )


def iter_consac_payloads(root: str | Path):
    base = Path(root)
    payload_dir = base / "payloads" if (base / "payloads").exists() else base
    yield from sorted(payload_dir.glob("*.npz"))
