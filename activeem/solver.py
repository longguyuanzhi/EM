from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np

from .benchmarks import BenchmarkSpec, synthetic_response


class SolverAdapter:
    def evaluate(self, x_norm: np.ndarray, spec: BenchmarkSpec) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class SyntheticEMSolver(SolverAdapter):
    """Fast analytic solver for smoke tests."""

    def __init__(self, noise_std: float = 0.0):
        self.noise_std = float(noise_std)

    def evaluate(self, x_norm: np.ndarray, spec: BenchmarkSpec) -> Dict[str, np.ndarray]:
        return synthetic_response(x_norm, spec, noise_std=self.noise_std)


class ExternalFEMSolver(SolverAdapter):
    """External solver wrapper.

    The external command should accept:
        command request.json response.npz

    The response NPZ must contain:
        spectra: [N,P,F]
        scalars: [N,S]
    """

    def __init__(self, command: str, workdir: str | Path = "runs/external_solver"):
        if not command:
            raise ValueError("ExternalFEMSolver requires a non-empty command.")
        self.command = command
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, x_norm: np.ndarray, spec: BenchmarkSpec) -> Dict[str, np.ndarray]:
        x_norm = np.atleast_2d(x_norm).astype(np.float32)
        req = self.workdir / "request.json"
        resp = self.workdir / "response.npz"
        payload = {
            "benchmark": spec.name,
            "x_norm": x_norm.tolist(),
            "freq_grid_ghz": spec.freq_grid.tolist(),
            "variables": [{"name": v.name, "low": v.low, "high": v.high, "unit": v.unit} for v in spec.variables],
        }
        req.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        cmd = self.command.format(request=str(req), response=str(resp))
        subprocess.run(cmd, shell=True, check=True)
        obj = np.load(resp)
        return {"x": x_norm, "freq": spec.freq_grid.astype(np.float32), "spectra": obj["spectra"].astype(np.float32), "scalars": obj["scalars"].astype(np.float32)}
