from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def combined_metrics(true: Dict[str, np.ndarray], pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "spectra_mae": mae(true["spectra"], pred["spectra"]),
        "spectra_rmse": rmse(true["spectra"], pred["spectra"]),
        "scalar_mae": mae(true["scalars"], pred["scalars"]),
        "scalar_rmse": rmse(true["scalars"], pred["scalars"]),
        "mae": mae(np.concatenate([true["spectra"].ravel(), true["scalars"].ravel()]), np.concatenate([pred["spectra"].ravel(), pred["scalars"].ravel()])),
        "rmse": rmse(np.concatenate([true["spectra"].ravel(), true["scalars"].ravel()]), np.concatenate([pred["spectra"].ravel(), pred["scalars"].ravel()])),
    }


def calibration_metrics(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, high_error_percentile: float = 80.0) -> Dict[str, float]:
    abs_err = np.abs(y_true - mean).reshape(y_true.shape[0], -1).mean(axis=1)
    unc = std.reshape(std.shape[0], -1).mean(axis=1)
    coverage = float(np.mean(np.abs(y_true - mean) <= 1.645 * std))
    order = np.argsort(unc)
    bins = np.array_split(order, 10)
    uce = 0.0
    for b in bins:
        if len(b):
            uce += len(b) / len(unc) * abs(abs_err[b].mean() - unc[b].mean())
    labels = abs_err >= np.percentile(abs_err, high_error_percentile)
    try:
        auroc = roc_auc_score(labels.astype(int), unc)
    except ValueError:
        auroc = float("nan")
    return {"uce": float(uce), "coverage90": coverage, "spearman": float(spearmanr(unc, abs_err).correlation), "auroc_high_error": float(auroc)}
