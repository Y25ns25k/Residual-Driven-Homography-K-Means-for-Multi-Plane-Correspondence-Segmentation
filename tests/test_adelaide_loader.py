from __future__ import annotations

import numpy as np
from scipy.io import savemat

from src.datasets.adelaide_loader import AdelaideLoader


def _payload(data: np.ndarray) -> dict[str, np.ndarray]:
    n = data.shape[0] if data.shape[1] >= 6 else data.shape[1]
    return {
        "img1": np.zeros((10, 12, 3), dtype=np.uint8),
        "img2": np.zeros((10, 12, 3), dtype=np.uint8),
        "data": data,
        "score": np.arange(n, dtype=np.float64).reshape(1, -1),
        "label": np.array([[1, 0, 2]], dtype=np.uint8),
    }


def test_adelaide_loader_reads_n_by_6_data(tmp_path) -> None:
    data = np.array(
        [
            [1, 2, 1, 11, 12, 1],
            [3, 4, 1, 13, 14, 1],
            [5, 6, 1, 15, 16, 1],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "toy.mat"
    savemat(path, _payload(data))

    sample = AdelaideLoader(tmp_path).load_file(path)

    assert sample.src_pts.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert sample.dst_pts.tolist() == [[11, 12], [13, 14], [15, 16]]
    assert sample.labels.tolist() == [1, 0, 2]
    assert sample.n_structures == 2
    assert sample.n_gross_outliers == 1


def test_adelaide_loader_reads_6_by_n_data(tmp_path) -> None:
    data = np.array(
        [
            [1, 3, 5],
            [2, 4, 6],
            [1, 1, 1],
            [11, 13, 15],
            [12, 14, 16],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "toy_transposed.mat"
    savemat(path, _payload(data))

    sample = AdelaideLoader(tmp_path).load_file(path)

    assert sample.src_pts.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert sample.dst_pts.tolist() == [[11, 12], [13, 14], [15, 16]]
