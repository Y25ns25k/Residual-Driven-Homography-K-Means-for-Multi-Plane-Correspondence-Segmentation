from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class AdelaideScene:
    scene_id: str
    x1: np.ndarray
    x2: np.ndarray
    labels: np.ndarray
    image_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class AdelaideLoadReport:
    scenes: list[AdelaideScene]
    skipped: list[tuple[str, str]]
    files_found: int


ADELAIDE_HOMOGRAPHY_SCENES = frozenset(
    {
        "barrsmith",
        "johnsona",
        "oldclassicswing",
        "johnsonb",
        "physics",
        "ladysymon",
        "sene",
        "elderhalla",
        "library",
        "elderhallb",
        "napiera",
        "unihouse",
        "bonhall",
        "napierb",
        "unionhouse",
        "bonython",
        "neem",
        "hartley",
        "nese",
    }
)

ADELAIDE_FUNDAMENTAL_SCENES = frozenset(
    {
        "breadcartoychips",
        "cubechips",
        "biscuit",
        "breadcube",
        "cubetoy",
        "biscuitbook",
        "breadcubechips",
        "dinobooks",
        "biscuitbookbox",
        "breadtoy",
        "toycubecar",
        "boardgame",
        "breadtoycar",
        "carchipscube",
        "game",
        "cube",
        "gamebiscuit",
        "book",
        "cubebreadtoychips",
    }
)


def adelaide_subset_names(subset: str) -> frozenset[str] | None:
    subset_l = subset.lower()
    if subset_l == "all":
        return None
    if subset_l == "homography":
        return ADELAIDE_HOMOGRAPHY_SCENES
    if subset_l == "fundamental":
        return ADELAIDE_FUNDAMENTAL_SCENES
    raise ValueError(f"unknown AdelaideRMF subset: {subset}")


def filter_adelaide_scenes(scenes: list[AdelaideScene], subset: str) -> tuple[list[AdelaideScene], list[str]]:
    names = adelaide_subset_names(subset)
    if names is None:
        return list(scenes), []
    by_id = {scene.scene_id.lower(): scene for scene in scenes}
    selected = [by_id[name] for name in sorted(names) if name in by_id]
    missing = sorted(name for name in names if name not in by_id)
    return selected, missing


def inspect_mat_file(path: str | Path) -> dict[str, tuple[int, ...]]:
    p = Path(path)
    try:
        payload = loadmat(p)
        return {k: tuple(np.asarray(v).shape) for k, v in payload.items() if not k.startswith("__")}
    except NotImplementedError:
        with h5py.File(p, "r") as f:
            return {k: tuple(v.shape) for k, v in f.items()}


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    try:
        raw = loadmat(path)
        return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.asarray(f[k]).T
        return out


def parse_adelaide_mat(path: str | Path) -> AdelaideScene:
    p = Path(path)
    payload = _load_payload(p)
    label_key = next((k for k in ["label", "labels", "Label", "Labels", "gt", "ground_truth"] if k in payload), None)
    data_key = next((k for k in ["data", "X", "matches", "points", "pts"] if k in payload), None)
    if label_key is None or data_key is None:
        raise ValueError(f"{p.name}: could not find data/label keys; available {inspect_mat_file(p)}")
    labels_raw = np.asarray(payload[label_key]).reshape(-1).astype(np.int32)
    labels = labels_raw.copy()
    labels[labels_raw == 0] = -1
    labels[labels_raw > 0] = labels_raw[labels_raw > 0] - 1
    data = np.asarray(payload[data_key], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{p.name}: data must be 2D")
    n = len(labels)
    if data.shape[0] == n and data.shape[1] >= 5:
        arr = data
    elif data.shape[1] == n and data.shape[0] >= 5:
        arr = data.T
    elif data.shape[0] == n and data.shape[1] >= 4:
        arr = data
    elif data.shape[1] == n and data.shape[0] >= 4:
        arr = data.T
    else:
        raise ValueError(f"{p.name}: data shape {data.shape} incompatible with labels {labels.shape}")
    x1 = arr[:, 0:2].copy()
    if arr.shape[1] >= 6 and np.allclose(arr[:, 2], 1.0, atol=1e-8):
        x2 = arr[:, 3:5].copy()
    elif arr.shape[1] >= 5:
        x2 = arr[:, 3:5].copy()
    else:
        x2 = arr[:, 2:4].copy()
    image_shape = None
    if "img1" in payload and payload["img1"].ndim >= 2:
        image_shape = tuple(payload["img1"].shape[:2])
    return AdelaideScene(p.stem, x1, x2, labels, image_shape)


def load_adelaide_directory_report(root: str | Path) -> AdelaideLoadReport:
    base = Path(root)
    if not base.exists():
        print(f"[adelaide] data directory does not exist: {base}")
        return AdelaideLoadReport([], [], 0)
    files = sorted(base.rglob("*.mat"))
    print(f"[adelaide] found {len(files)} .mat files under {base}")
    scenes: list[AdelaideScene] = []
    skipped: list[tuple[str, str]] = []
    for path in files:
        try:
            scenes.append(parse_adelaide_mat(path))
        except Exception as exc:
            reason = str(exc)
            skipped.append((str(path), reason))
            print(f"[adelaide] skipping {path}: {reason}")
    print(f"[adelaide] parsed {len(scenes)} scenes, skipped {len(skipped)} files")
    return AdelaideLoadReport(scenes, skipped, len(files))


def load_adelaide_directory(root: str | Path) -> list[AdelaideScene]:
    return load_adelaide_directory_report(root).scenes
