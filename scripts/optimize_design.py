from __future__ import annotations

import argparse

from activeem.acquisition import AcquisitionConfig
from activeem.active_loop import ActiveLearningRunner, TrainConfig
from activeem.benchmarks import get_benchmark_spec
from activeem.optimize import random_search_optimize, verify_design
from activeem.solver import SyntheticEMSolver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["filter", "resonator", "coupler"], required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    spec = get_benchmark_spec(args.benchmark)
    solver = SyntheticEMSolver()
    runner = ActiveLearningRunner(spec, solver, TrainConfig(epochs=40, ensemble_size=2, mc_samples=2), AcquisitionConfig(), args.seed)
    data = runner.initialize(n_initial=30, n_val=40, n_test=40)
    models = runner.train_ensemble(data["train"], data["val"], 0)
    best = random_search_optimize(models, spec, n_candidates=1000, seed=args.seed)
    verified = verify_design(solver, spec, best["x"])
    print("Best normalized x:", best["x"])
    print("Predicted scalars:", best["pred_scalars"])
    print("Verified scalars:", verified["scalars"][0])


if __name__ == "__main__":
    main()
