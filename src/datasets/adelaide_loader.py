from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class AdelaideSample:
    name: str
    path: Path
    img1: np.ndarray
    img2: np.ndarray
    src_pts: np.ndarray     # (N, 2) first-image coordinates
    dst_pts: np.ndarray     # (N, 2) second-image coordinates
    labels: np.ndarray      # (N,) int — 0 = gross outlier, 1..K = structure
    score: np.ndarray       # (N,) match scores (may be all-1)

    @property
    def n_points(self) -> int:
        return int(len(self.labels))

    @property
    def structure_labels(self) -> np.ndarray:
        """Sorted unique structure label values (excluding outlier=0)."""
        return np.asarray(
            sorted(int(x) for x in np.unique(self.labels) if int(x) > 0),
            dtype=np.int32,
        )

    @property
    def n_structures(self) -> int:
        return int(len(self.structure_labels))

    @property
    def n_gross_outliers(self) -> int:
        return int(np.sum(self.labels == 0))

    @property
    def gross_outlier_rate(self) -> float:
        return float(self.n_gross_outliers / max(1, self.n_points))


class AdelaideLoader:
    """Load AdelaideRMF .mat correspondence files.

    Each file stores (6, N) data matrix [x1; y1; 1; x2; y2; 1] and a label
    vector (N,) where 0 = gross outlier and 1..K = structure index.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def available(self) -> bool:
        return self.root.exists() and any(self.root.glob("*.mat"))

    def files(self) -> List[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.mat"))

    def iter_samples(self) -> Iterable[AdelaideSample]:
        for path in self.files():
            yield self.load_file(path)

    def load_file(self, path: str | Path) -> AdelaideSample:
        mat_path = Path(path)
        payload = loadmat(mat_path)
        required = ("data", "label")
        missing = [k for k in required if k not in payload]
        if missing:
            raise KeyError(f"{mat_path.name} missing keys: {missing}")

        labels = np.asarray(payload["label"]).reshape(-1).astype(np.int32)
        data = self._normalize(np.asarray(payload["data"], dtype=np.float64), len(labels), mat_path)
        src_pts = data[:, 0:2].copy()
        dst_pts = data[:, 3:5].copy()

        # Images (optional — some versions of the dataset omit them)
        img1 = np.asarray(payload["img1"]) if "img1" in payload else np.zeros((1, 1, 3), dtype=np.uint8)
        img2 = np.asarray(payload["img2"]) if "img2" in payload else np.zeros((1, 1, 3), dtype=np.uint8)
        score = np.asarray(payload["score"]).reshape(-1) if "score" in payload else np.ones(len(labels))

        return AdelaideSample(
            name=mat_path.stem,
            path=mat_path,
            img1=img1,
            img2=img2,
            src_pts=src_pts,
            dst_pts=dst_pts,
            labels=labels,
            score=score,
        )

    @staticmethod
    def _normalize(data: np.ndarray, n: int, path: Path) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError(f"{path.name} data must be 2D, got {data.shape}")
        if data.shape[0] == n and data.shape[1] >= 5:
            return data
        if data.shape[1] == n and data.shape[0] >= 5:
            return data.T
        raise ValueError(
            f"{path.name} data shape {data.shape} incompatible with {n} labels"
        )
