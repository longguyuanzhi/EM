from __future__ import annotations

from typing import Dict, List

import numpy as np

from .benchmarks import BenchmarkSpec, lhs_sample
from .solver import SolverAdapter
from .uncertainty import ensemble_predict


def default_objective(spec: BenchmarkSpec, mean_spectra: np.ndarray, mean_scalars: np.ndarray) -> np.ndarray:
    if spec.name == "filter":
        return np.abs(mean_scalars[:, 0] - 2.45) + 0.05 * mean_scalars[:, 2]
    if spec.name == "resonator":
        return np.abs(mean_scalars[:, 0] - 5.00) - 0.01 * mean_scalars[:, 1]
    if spec.name == "coupler":
        return np.abs(mean_scalars[:, 0] + 3.0) - 0.02 * mean_scalars[:, 1]
    return mean_scalars.mean(axis=1)


def random_search_optimize(models: List, spec: BenchmarkSpec, n_candidates: int = 5000, seed: int = 0, device: str = "cpu") -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = lhs_sample(n_candidates, spec.dim, rng)
    stats = ensemble_predict(models, x, spec.freq_grid, device=device, mc_samples=4)
    obj = default_objective(spec, stats.mean_spectra, stats.mean_scalars)
    best = int(np.argmin(obj))
    return {"x": x[best], "objective": obj[best], "pred_scalars": stats.mean_scalars[best], "pred_spectra": stats.mean_spectra[best]}


def verify_design(solver: SolverAdapter, spec: BenchmarkSpec, x: np.ndarray) -> Dict[str, np.ndarray]:
    return solver.evaluate(np.atleast_2d(x), spec)
