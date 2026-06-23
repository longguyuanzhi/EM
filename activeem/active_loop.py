from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .acquisition import AcquisitionConfig, select_batch
from .benchmarks import BenchmarkSpec, lhs_sample
from .data import EMDataset, concat_data
from .losses import total_loss
from .metrics import combined_metrics
from .models import FrequencyConditionedFNO, MLPBaseline
from .solver import SolverAdapter
from .uncertainty import TemperatureCalibrator, ensemble_predict
from .utils import ensure_dir, save_json, set_seed


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    lambda_phy: float = 0.1
    lambda_reg: float = 0.0
    ensemble_size: int = 3
    mc_samples: int = 4
    device: str = "cpu"
    model_type: str = "fno"


def build_model(spec: BenchmarkSpec, cfg: TrainConfig, seed: int) -> torch.nn.Module:
    set_seed(seed)
    if cfg.model_type == "mlp":
        return MLPBaseline(spec.dim, len(spec.spectral_ports), spec.n_freq, len(spec.scalar_targets))
    return FrequencyConditionedFNO(spec.dim, len(spec.spectral_ports), len(spec.scalar_targets), spec.name, spec.n_freq)


def train_one_model(model: torch.nn.Module, train_data: Dict[str, np.ndarray], val_data: Optional[Dict[str, np.ndarray]], spec: BenchmarkSpec, cfg: TrainConfig) -> torch.nn.Module:
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(EMDataset(train_data["x"], train_data["freq"], train_data["spectra"], train_data["scalars"]), batch_size=cfg.batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    bad = 0
    patience = 20

    for _ in range(cfg.epochs):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred = model(batch["x"], batch["freq"])
            loss = total_loss(pred, batch, spec.name, cfg.lambda_phy, cfg.lambda_reg, model)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                xb = torch.as_tensor(val_data["x"], dtype=torch.float32, device=device)
                fr = torch.as_tensor(val_data["freq"], dtype=torch.float32, device=device)
                pred = model(xb, fr)
                val_metric = torch.mean(torch.abs(pred["spectra"].cpu() - torch.as_tensor(val_data["spectra"]))).item()
            if val_metric < best_val:
                best_val = val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class ActiveLearningRunner:
    """Closed-loop active surrogate learner."""

    def __init__(self, spec: BenchmarkSpec, solver: SolverAdapter, train_cfg: TrainConfig, acq_cfg: AcquisitionConfig, seed: int = 0, out_dir: str | Path = "runs/activeem"):
        self.spec = spec
        self.solver = solver
        self.train_cfg = train_cfg
        self.acq_cfg = acq_cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.out_dir = ensure_dir(out_dir)

    def initialize(self, n_initial: Optional[int] = None, n_val: int = 80, n_test: int = 200) -> Dict[str, Dict[str, np.ndarray]]:
        n_initial = n_initial or self.spec.initial_samples
        x0 = lhs_sample(n_initial, self.spec.dim, self.rng)
        xv = lhs_sample(n_val, self.spec.dim, self.rng)
        xt = lhs_sample(n_test, self.spec.dim, self.rng)
        return {"train": self.solver.evaluate(x0, self.spec), "val": self.solver.evaluate(xv, self.spec), "test": self.solver.evaluate(xt, self.spec)}

    def train_ensemble(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray], round_id: int) -> List[torch.nn.Module]:
        models = []
        for m in range(self.train_cfg.ensemble_size):
            model = build_model(self.spec, self.train_cfg, self.seed + 1000 * round_id + m)
            models.append(train_one_model(model, train_data, val_data, self.spec, self.train_cfg))
        return models

    def evaluate_models(self, models: List[torch.nn.Module], data: Dict[str, np.ndarray]) -> Dict[str, float]:
        stats = ensemble_predict(models, data["x"], data["freq"], self.train_cfg.device, self.train_cfg.mc_samples)
        return combined_metrics(data, {"spectra": stats.mean_spectra, "scalars": stats.mean_scalars})

    def run(self, rounds: int = 3, initial: Optional[int] = None, batch_size: Optional[int] = None, candidate_pool: Optional[int] = None, n_val: int = 80, n_test: int = 200) -> Dict[str, object]:
        batch_size = batch_size or self.spec.active_batch
        candidate_pool = candidate_pool or self.spec.candidate_pool
        data = self.initialize(initial, n_val, n_test)
        history = []

        for t in range(rounds):
            models = self.train_ensemble(data["train"], data["val"], t)
            test_metrics = self.evaluate_models(models, data["test"])
            print(f"[round {t}] n={data['train']['x'].shape[0]} test_MAE={test_metrics['mae']:.4f}")
            row = {"round": t, "n_train": int(data["train"]["x"].shape[0]), **test_metrics}

            x_cand = lhs_sample(candidate_pool, self.spec.dim, self.rng)
            cand_stats = ensemble_predict(models, x_cand, self.spec.freq_grid, self.train_cfg.device, self.train_cfg.mc_samples)
            val_stats = ensemble_predict(models, data["val"]["x"], data["val"]["freq"], self.train_cfg.device, self.train_cfg.mc_samples)

            cal = TemperatureCalibrator().fit(val_stats.mean_spectra, val_stats.std_spectra, data["val"]["spectra"], val_stats.mean_scalars, val_stats.std_scalars, data["val"]["scalars"])
            cand_stats = cal.transform(cand_stats)
            val_stats_cal = cal.transform(val_stats)

            val_err = np.abs(data["val"]["spectra"] - val_stats.mean_spectra).mean(axis=(1, 2))
            val_unc = val_stats_cal.std_spectra.mean(axis=(1, 2))
            acq = select_batch(x_cand, cand_stats, data["train"]["x"], batch_size, self.acq_cfg, val_uncertainty=val_unc, val_error=val_err, emax=self.spec.val_mae_stop)

            new_data = self.solver.evaluate(acq["x"], self.spec)
            data["train"] = concat_data(data["train"], new_data)
            row.update({"safe_fraction": acq["safe_fraction"], "safe_threshold": acq["safe_threshold"]})
            history.append(row)

        models = self.train_ensemble(data["train"], data["val"], rounds)
        final_metrics = self.evaluate_models(models, data["test"])
        result = {
            "benchmark": self.spec.name,
            "seed": self.seed,
            "n_train_final": int(data["train"]["x"].shape[0]),
            "history": history,
            "final_metrics": final_metrics,
            "train_config": asdict(self.train_cfg),
            "acquisition_config": asdict(self.acq_cfg),
        }
        save_json(result, self.out_dir / "results.json")
        return result
