from __future__ import annotations

import argparse

from activeem.acquisition import AcquisitionConfig
from activeem.active_loop import ActiveLearningRunner, TrainConfig
from activeem.benchmarks import get_benchmark_spec
from activeem.solver import SyntheticEMSolver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["filter", "resonator", "coupler"], required=True)
    p.add_argument("--method", choices=["mlp", "fno"], default="mlp")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/baseline")
    args = p.parse_args()

    spec = get_benchmark_spec(args.benchmark)
    runner = ActiveLearningRunner(spec, SyntheticEMSolver(), TrainConfig(model_type=args.method, ensemble_size=2, mc_samples=2, epochs=40), AcquisitionConfig(), args.seed, args.out)
    print(runner.run(rounds=args.rounds, initial=20, batch_size=5, candidate_pool=300)["final_metrics"])


if __name__ == "__main__":
    main()
