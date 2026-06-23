from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class VariableSpec:
    name: str
    meaning: str
    low: float
    high: float
    unit: str = "mm"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    structure: str
    material_stack: str
    variables: List[VariableSpec]
    freq_start_ghz: float
    freq_stop_ghz: float
    n_freq: int
    spectral_ports: Tuple[str, ...]
    scalar_targets: Tuple[str, ...]
    initial_samples: int
    active_batch: int
    final_samples: int
    candidate_pool: int
    val_mae_stop: float

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def freq_grid(self) -> np.ndarray:
        return np.linspace(self.freq_start_ghz, self.freq_stop_ghz, self.n_freq, dtype=np.float32)

    @property
    def lows(self) -> np.ndarray:
        return np.array([v.low for v in self.variables], dtype=np.float32)

    @property
    def highs(self) -> np.ndarray:
        return np.array([v.high for v in self.variables], dtype=np.float32)


def list_benchmarks() -> List[str]:
    return ["filter", "resonator", "coupler"]


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    key = name.lower()
    if key == "filter":
        return BenchmarkSpec(
            name="filter",
            structure="Two-pole microstrip bandpass filter with coupled resonators and tapped feed lines.",
            material_stack="Rogers RO4003C-like substrate, eps_r=3.55, tan_delta=0.0027, h=0.813 mm, copper=0.035 mm.",
            variables=[
                VariableSpec("L1", "First resonator-section length", 15.0, 19.0),
                VariableSpec("L2", "Second resonator-section length", 14.0, 18.0),
                VariableSpec("g12", "Inter-resonator coupling gap", 0.20, 1.20),
                VariableSpec("Wf", "Feed-line width", 1.40, 2.80),
                VariableSpec("sf", "Feed tap/offset distance", 0.30, 1.50),
            ],
            freq_start_ghz=1.8,
            freq_stop_ghz=3.2,
            n_freq=201,
            spectral_ports=("S11", "S21"),
            scalar_targets=("f0", "bandwidth_3db", "insertion_loss_db"),
            initial_samples=50,
            active_batch=10,
            final_samples=150,
            candidate_pool=5000,
            val_mae_stop=0.020,
        )
    if key == "resonator":
        return BenchmarkSpec(
            name="resonator",
            structure="Rectangular substrate-integrated/cavity resonator with tunable slot and feed offset.",
            material_stack="Dielectric cavity substrate, eps_r in [3.38,3.75], tan_delta=0.002-0.004, finite-conductivity copper.",
            variables=[
                VariableSpec("Lc", "Cavity length", 18.0, 26.0),
                VariableSpec("Wc", "Cavity width", 12.0, 20.0),
                VariableSpec("ls", "Slot length", 4.0, 9.0),
                VariableSpec("ws", "Slot width", 0.20, 1.20),
                VariableSpec("of", "Feed offset", 1.0, 5.0),
                VariableSpec("eps_r", "Effective dielectric constant", 3.38, 3.75, unit=""),
            ],
            freq_start_ghz=4.0,
            freq_stop_ghz=8.0,
            n_freq=401,
            spectral_ports=("S11",),
            scalar_targets=("f0", "Q", "min_S11_db"),
            initial_samples=60,
            active_batch=10,
            final_samples=180,
            candidate_pool=6000,
            val_mae_stop=0.022,
        )
    if key == "coupler":
        return BenchmarkSpec(
            name="coupler",
            structure="Four-port branch-line/directional coupler with tunable branch lengths, widths, and coupling gap.",
            material_stack="Rogers RO4350B-like substrate, eps_r=3.48, tan_delta=0.0037, h=0.762 mm, copper=0.035 mm.",
            variables=[
                VariableSpec("Lm1", "Main-line length 1", 15.0, 22.0),
                VariableSpec("Lm2", "Main-line length 2", 15.0, 22.0),
                VariableSpec("Wm", "Main-line width", 0.80, 2.40),
                VariableSpec("Lb1", "Branch-line length 1", 10.0, 18.0),
                VariableSpec("Lb2", "Branch-line length 2", 10.0, 18.0),
                VariableSpec("Wb", "Branch-line width", 0.50, 2.00),
                VariableSpec("gc", "Coupling gap", 0.15, 1.00),
                VariableSpec("tf", "Feed taper length", 0.50, 3.00),
            ],
            freq_start_ghz=2.0,
            freq_stop_ghz=4.0,
            n_freq=201,
            spectral_ports=("S11", "S21", "S31", "S41"),
            scalar_targets=("coupling_db", "isolation_db", "insertion_loss_db", "phase_imbalance_deg"),
            initial_samples=60,
            active_batch=10,
            final_samples=170,
            candidate_pool=6000,
            val_mae_stop=0.021,
        )
    raise ValueError(f"Unknown benchmark {name!r}. Valid options: {list_benchmarks()}")


def lhs_sample(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube samples in [-1,1]^dim."""
    u = np.zeros((n, dim), dtype=np.float32)
    for j in range(dim):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return (2.0 * u - 1.0).astype(np.float32)


def denormalize_x(x_norm: np.ndarray, spec: BenchmarkSpec) -> np.ndarray:
    x_norm = np.asarray(x_norm, dtype=np.float32)
    return spec.lows + 0.5 * (x_norm + 1.0) * (spec.highs - spec.lows)


def normalize_x(x_phys: np.ndarray, spec: BenchmarkSpec) -> np.ndarray:
    x_phys = np.asarray(x_phys, dtype=np.float32)
    return 2.0 * (x_phys - spec.lows) / (spec.highs - spec.lows) - 1.0


def synthetic_response(x_norm: np.ndarray, spec: BenchmarkSpec, noise_std: float = 0.0) -> Dict[str, np.ndarray]:
    """EM-like analytic response generator for smoke tests only."""
    x = np.atleast_2d(x_norm).astype(np.float32)
    f = spec.freq_grid[None, :]
    rng = np.random.default_rng(12345)

    if spec.name == "filter":
        f0 = 2.45 + 0.10 * x[:, 0:1] - 0.07 * x[:, 1:2] + 0.04 * np.sin(np.pi * x[:, 4:5])
        bw = 0.28 + 0.05 * (1 - x[:, 2:3]) + 0.03 * np.cos(np.pi * x[:, 3:4])
        band = np.exp(-((f - f0) / bw) ** 2)
        s21 = np.clip(0.12 + 0.82 * band - 0.05 * np.abs(x[:, 2:3]), 0, 1.2)
        s11 = np.clip(0.90 - 0.75 * band + 0.08 * np.abs(x[:, 3:4] - 0.2), 0, 1.2)
        spectra = np.stack([s11, s21], axis=1)
        il_db = -20 * np.log10(np.maximum(np.max(s21, axis=1), 1e-4))
        scalars = np.concatenate([f0, bw, il_db[:, None]], axis=1)
    elif spec.name == "resonator":
        f0 = 5.0 + 0.45 * x[:, 0:1] - 0.35 * x[:, 1:2] - 0.18 * x[:, 5:6]
        q = 900 + 120 * (1 - x[:, 3:4]) + 50 * np.cos(np.pi * x[:, 4:5])
        gamma = np.clip(f0 / q, 0.005, 0.08)
        dip = np.exp(-((f - f0) / (8 * gamma)) ** 2)
        s11 = np.clip(0.95 - 0.82 * dip + 0.05 * np.abs(x[:, 2:3]), 0, 1.2)
        spectra = np.expand_dims(s11, axis=1)
        min_s11_db = 20 * np.log10(np.maximum(np.min(s11, axis=1), 1e-4))
        scalars = np.concatenate([f0, q / 1000.0, min_s11_db[:, None] / 30.0], axis=1)
    elif spec.name == "coupler":
        f0 = 3.0 + 0.12 * x[:, 0:1] - 0.10 * x[:, 1:2] + 0.05 * x[:, 7:8]
        width = 0.32 + 0.04 * np.abs(x[:, 6:7])
        band = np.exp(-((f - f0) / width) ** 2)
        s11 = np.clip(0.35 - 0.22 * band + 0.05 * np.abs(x[:, 2:3]), 0, 1.2)
        s21 = np.clip(0.82 - 0.18 * band + 0.03 * x[:, 5:6], 0, 1.2)
        s31 = np.clip(0.68 * band + 0.02 * x[:, 3:4], 0, 1.2)
        s41 = np.clip(0.08 + 0.04 * np.abs(x[:, 4:5]) + 0.04 * (1 - band), 0, 1.2)
        spectra = np.stack([s11, s21, s31, s41], axis=1)
        coupling_db = 20 * np.log10(np.maximum(np.max(s31, axis=1), 1e-4))
        isolation_db = -20 * np.log10(np.maximum(np.min(s41, axis=1), 1e-4))
        il_db = -20 * np.log10(np.maximum(np.max(s21, axis=1), 1e-4))
        phase = 2.0 + 4.0 * np.abs(x[:, 0] - x[:, 1])
        scalars = np.stack([coupling_db, isolation_db / 30.0, il_db, phase / 10.0], axis=1)
    else:
        raise ValueError(spec.name)

    if noise_std > 0:
        spectra = spectra + rng.normal(0, noise_std, spectra.shape).astype(np.float32)
        scalars = scalars + rng.normal(0, noise_std, scalars.shape).astype(np.float32)

    return {"x": x.astype(np.float32), "freq": spec.freq_grid.astype(np.float32), "spectra": spectra.astype(np.float32), "scalars": scalars.astype(np.float32)}
