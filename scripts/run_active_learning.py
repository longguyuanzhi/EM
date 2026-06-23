from __future__ import annotations

import argparse

from activeem.acquisition import AcquisitionConfig
from activeem.active_loop import ActiveLearningRunner, TrainConfig
from activeem.benchmarks import get_benchmark_spec
from activeem.solver import ExternalFEMSolver, SyntheticEMSolver
from activeem.utils import load_yaml, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--initial", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--candidate-pool", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--solver-command", default="")
    p.add_argument("--out", default="runs/activeem")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    spec = get_benchmark_spec(cfg["benchmark"])
    set_seed(args.seed)
    solver = SyntheticEMSolver() if args.synthetic else ExternalFEMSolver(args.solver_command)
    runner = ActiveLearningRunner(spec, solver, TrainConfig(device=args.device), AcquisitionConfig(), seed=args.seed, out_dir=args.out)
    result = runner.run(rounds=args.rounds, initial=args.initial, batch_size=args.batch, candidate_pool=args.candidate_pool)
    print(result["final_metrics"])


if __name__ == "__main__":
    main()
