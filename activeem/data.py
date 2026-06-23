from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .benchmarks import BenchmarkSpec, lhs_sample, synthetic_response


class EMDataset(Dataset):
    """Torch dataset for frequency-domain EM responses."""

    def __init__(self, x: np.ndarray, freq: np.ndarray, spectra: np.ndarray, scalars: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.freq = torch.as_tensor(freq, dtype=torch.float32)
        self.spectra = torch.as_tensor(spectra, dtype=torch.float32)
        self.scalars = torch.as_tensor(scalars, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[idx], "freq": self.freq, "spectra": self.spectra[idx], "scalars": self.scalars[idx]}


def save_npz(path: str | Path, data: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    obj = np.load(path)
    return {k: obj[k] for k in obj.files}


def split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def make_synthetic_dataset(spec: BenchmarkSpec, n: int, seed: int, noise_std: float = 0.0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = lhs_sample(n, spec.dim, rng)
    return synthetic_response(x, spec, noise_std=noise_std)


def subset(data: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: (v if k == "freq" else v[idx]) for k, v in data.items()}


def concat_data(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "x": np.concatenate([a["x"], b["x"]], axis=0),
        "freq": a["freq"],
        "spectra": np.concatenate([a["spectra"], b["spectra"]], axis=0),
        "scalars": np.concatenate([a["scalars"], b["scalars"]], axis=0),
    }
