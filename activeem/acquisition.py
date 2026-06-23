from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .uncertainty import PredictionStats, aggregate_uncertainty


@dataclass
class AcquisitionConfig:
    alpha: float = 0.50
    beta: float = 0.25
    gamma: float = 0.25
    k_neighbors: int = 8
    residual_weight: float = 0.5
    diversity_eta: float = 0.7
    diversity_threshold: float = 0.05
    safe_delta: float = 0.05
    safe_quantile_fallback: float = 0.90


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def objective_vector(stats: PredictionStats) -> np.ndarray:
    if stats.mean_scalars.ndim == 2 and stats.mean_scalars.shape[1] > 0:
        return stats.mean_scalars
    return np.stack([stats.mean_spectra.mean(axis=(1, 2)), stats.mean_spectra.min(axis=(1, 2))], axis=1)


def nondominated(points: np.ndarray, minimize: Optional[np.ndarray] = None) -> np.ndarray:
    points = np.asarray(points)
    n, d = points.shape
    if minimize is None:
        minimize = np.ones(d, dtype=bool)
    signed = points.copy()
    signed[:, ~minimize] *= -1.0
    out = np.ones(n, dtype=bool)
    for i in range(n):
        dominated_by_any = np.all(signed <= signed[i], axis=1) & np.any(signed < signed[i], axis=1)
        if np.any(dominated_by_any):
            out[i] = False
    return out


def pareto_score(obj: np.ndarray, ref_obj: np.ndarray) -> np.ndarray:
    if ref_obj.size == 0:
        return np.ones(obj.shape[0])
    nd = ref_obj[nondominated(ref_obj)]
    dist = np.sqrt(((obj[:, None, :] - nd[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    return 1.0 / (dist + 1e-6)


def boundary_score(x_candidate: np.ndarray, stats: PredictionStats, x_labeled: np.ndarray, residual_labeled: Optional[np.ndarray] = None, k_neighbors: int = 8, residual_weight: float = 0.5) -> np.ndarray:
    obj = objective_vector(stats)
    nn_c = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(x_candidate))).fit(x_candidate)
    inds = nn_c.kneighbors(x_candidate, return_distance=False)[:, 1:]
    local_obj_std = np.array([obj[ii].std(axis=0).mean() for ii in inds])
    if residual_labeled is None or len(x_labeled) == 0:
        local_res = np.zeros(len(x_candidate))
    else:
        nn_l = NearestNeighbors(n_neighbors=min(k_neighbors, len(x_labeled))).fit(x_labeled)
        ii = nn_l.kneighbors(x_candidate, return_distance=False)
        local_res = residual_labeled[ii].mean(axis=1)
    return local_obj_std + residual_weight * local_res


def calibrated_safe_mask(uncertainty: np.ndarray, val_uncertainty: Optional[np.ndarray] = None, val_error: Optional[np.ndarray] = None, emax: Optional[float] = None, delta: float = 0.05, fallback_quantile: float = 0.90) -> Tuple[np.ndarray, float]:
    if val_uncertainty is not None and val_error is not None and emax is not None and len(val_uncertainty) >= 10:
        order = np.argsort(val_uncertainty)
        tau = float(np.quantile(val_uncertainty, fallback_quantile))
        for idx in order:
            cand_tau = val_uncertainty[idx]
            ok = val_uncertainty <= cand_tau
            if ok.sum() >= 5 and np.mean(val_error[ok] <= emax) >= 1.0 - delta:
                tau = float(cand_tau)
        return uncertainty <= tau, tau
    tau = float(np.quantile(uncertainty, fallback_quantile))
    return uncertainty <= tau, tau


def select_batch(x_candidate: np.ndarray, stats: PredictionStats, x_labeled: np.ndarray, batch_size: int, cfg: AcquisitionConfig, residual_labeled: Optional[np.ndarray] = None, val_uncertainty: Optional[np.ndarray] = None, val_error: Optional[np.ndarray] = None, emax: Optional[float] = None) -> dict:
    u = aggregate_uncertainty(stats)
    safe_mask, tau = calibrated_safe_mask(u, val_uncertainty, val_error, emax, cfg.safe_delta, cfg.safe_quantile_fallback)
    safe_idx = np.where(safe_mask)[0]
    if len(safe_idx) == 0:
        safe_idx = np.argsort(u)[:max(batch_size, 1)]

    xs = x_candidate[safe_idx]
    safe_stats = PredictionStats(stats.mean_spectra[safe_idx], stats.mean_scalars[safe_idx], stats.std_spectra[safe_idx], stats.std_scalars[safe_idx])
    us = u[safe_idx]
    bs = boundary_score(xs, safe_stats, x_labeled, residual_labeled, cfg.k_neighbors, cfg.residual_weight)
    obj = objective_vector(safe_stats)
    ps = pareto_score(obj, obj)

    score = cfg.alpha * minmax(us) + cfg.beta * minmax(bs) + cfg.gamma * minmax(ps)
    order = list(np.argsort(-score))
    selected = []
    selected_obj = []

    for idx in order:
        x = xs[idx]
        o = obj[idx]
        if not selected:
            selected.append(idx)
            selected_obj.append(o)
        else:
            dx = np.sqrt(((xs[selected] - x) ** 2).sum(axis=1))
            do = np.sqrt(((np.array(selected_obj) - o) ** 2).sum(axis=1))
            d = cfg.diversity_eta * dx + (1 - cfg.diversity_eta) * do
            if d.min() >= cfg.diversity_threshold:
                selected.append(idx)
                selected_obj.append(o)
        if len(selected) >= batch_size:
            break

    if len(selected) < batch_size:
        for idx in order:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= batch_size:
                break

    global_idx = safe_idx[np.array(selected, dtype=int)]
    return {"indices": global_idx, "x": x_candidate[global_idx], "scores": score[selected], "safe_threshold": tau, "safe_fraction": float(safe_mask.mean())}
