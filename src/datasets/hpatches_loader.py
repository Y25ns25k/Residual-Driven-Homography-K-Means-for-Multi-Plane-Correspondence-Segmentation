from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.io_utils import imread


@dataclass
class HPatchesScene:
    name: str
    scene_type: str   # 'v' (viewpoint) or 'i' (illumination)
    images: List[np.ndarray]
    homographies: List[np.ndarray]   # H_{1->k} for k=2..6


def _load_homography(path: Path) -> np.ndarray:
    h = np.loadtxt(str(path), dtype=np.float64)
    return h / h[2, 2]


def load_hpatches_scene(scene_dir: str | Path) -> HPatchesScene:
    scene_path = Path(scene_dir)
    name = scene_path.name
    scene_type = name[0] if name else "?"
    images: List[np.ndarray] = []
    homographies: List[np.ndarray] = []
    for idx in range(1, 7):
        img_path = scene_path / f"{idx}.ppm"
        if not img_path.exists():
            break
        images.append(imread(img_path, cv2.IMREAD_COLOR))
        if idx > 1:
            h_path = scene_path / f"H_1_{idx}"
            if h_path.exists():
                homographies.append(_load_homography(h_path))
            else:
                homographies.append(np.eye(3, dtype=np.float64))
    return HPatchesScene(name=name, scene_type=scene_type, images=images, homographies=homographies)


def list_hpatches_scenes(
    hpatches_dir: str | Path,
    scene_type: Optional[str] = None,
) -> List[Path]:
    root = Path(hpatches_dir)
    if not root.exists():
        return []
    scenes = sorted(p for p in root.iterdir() if p.is_dir())
    if scene_type is not None:
        scenes = [p for p in scenes if p.name.startswith(scene_type)]
    return scenes


def load_viewpoint_scenes(hpatches_dir: str | Path) -> List[HPatchesScene]:
    dirs = list_hpatches_scenes(hpatches_dir, scene_type="v")
    return [load_hpatches_scene(d) for d in dirs]
