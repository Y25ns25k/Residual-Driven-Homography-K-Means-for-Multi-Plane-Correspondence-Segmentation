from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path_obj)
    data = np.fromfile(str(path_obj), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"failed to decode image: {path_obj}")
    return image


def imwrite(path: str | Path, image: np.ndarray) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = path_obj.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"failed to encode image: {path_obj}")
    encoded.tofile(str(path_obj))


def read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: object, indent: int = 2) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path_obj.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
