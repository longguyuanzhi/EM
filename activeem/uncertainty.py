from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn


@dataclass
class PredictionStats:
    mean_spectra: np.ndarray
    mean_scalars: np.ndarray
    std_spectra: np.ndarray
    std_scalars: np.ndarray


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def ensemble_predict(models: List[nn.Module], x: np.ndarray, freq: np.ndarray, device: str = "cpu", mc_samples: int = 8, batch_size: int = 256) -> PredictionStats:
    """Predict mean/std with ensemble + MC dropout."""
    xs = torch.as_tensor(x, dtype=torch.float32)
    fr = torch.as_tensor(freq, dtype=torch.float32, device=device)
    all_s, all_c = [], []
    for model in models:
        model.to(device)
        model.eval()
        for _ in range(mc_samples):
            enable_dropout(model)
            chunks_s, chunks_c = [], []
            for st in range(0, xs.shape[0], batch_size):
                pred = model(xs[st:st + batch_size].to(device), fr)
                chunks_s.append(pred["spectra"].cpu())
                chunks_c.append(pred["scalars"].cpu())
            all_s.append(torch.cat(chunks_s, dim=0).numpy())
            all_c.append(torch.cat(chunks_c, dim=0).numpy())
    sp = np.stack(all_s, axis=0)
    sc = np.stack(all_c, axis=0)
    return PredictionStats(sp.mean(axis=0), sc.mean(axis=0), sp.std(axis=0, ddof=1) + 1e-8, sc.std(axis=0, ddof=1) + 1e-8)


@dataclass
class TemperatureCalibrator:
    spectra_scale: float = 1.0
    scalar_scale: float = 1.0

    def fit(self, mean_spectra: np.ndarray, std_spectra: np.ndarray, y_spectra: np.ndarray, mean_scalars: np.ndarray, std_scalars: np.ndarray, y_scalars: np.ndarray) -> "TemperatureCalibrator":
        eps = 1e-8
        self.spectra_scale = float(np.sqrt(np.mean((y_spectra - mean_spectra) ** 2) / (np.mean(std_spectra ** 2) + eps)))
        self.scalar_scale = float(np.sqrt(np.mean((y_scalars - mean_scalars) ** 2) / (np.mean(std_scalars ** 2) + eps)))
        self.spectra_scale = float(np.clip(self.spectra_scale, 0.1, 10.0))
        self.scalar_scale = float(np.clip(self.scalar_scale, 0.1, 10.0))
        return self

    def transform(self, stats: PredictionStats) -> PredictionStats:
        return PredictionStats(stats.mean_spectra, stats.mean_scalars, stats.std_spectra * self.spectra_scale, stats.std_scalars * self.scalar_scale)


def aggregate_uncertainty(stats: PredictionStats, scalar_weight: float = 0.2) -> np.ndarray:
    u_s = stats.std_spectra.mean(axis=(1, 2))
    u_c = stats.std_scalars.mean(axis=1) if stats.std_scalars.size else 0.0
    return u_s + scalar_weight * u_c
