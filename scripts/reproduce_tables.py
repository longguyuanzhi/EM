from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from activeem.acquisition import AcquisitionConfig
from activeem.active_loop import ActiveLearningRunner, TrainConfig
from activeem.benchmarks import get_benchmark_spec
from activeem.solver import SyntheticEMSolver
from activeem.utils import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="runs/rebuttal_tables_synthetic")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg_path in args.configs:
        cfg = load_yaml(cfg_path)
        spec = get_benchmark_spec(cfg["benchmark"])
        for seed in args.seeds:
            runner = ActiveLearningRunner(spec, SyntheticEMSolver(), TrainConfig(epochs=30, ensemble_size=2, mc_samples=2), AcquisitionConfig(), seed=seed, out_dir=out / f"{spec.name}_seed{seed}")
            result = runner.run(rounds=1, initial=20, batch_size=5, candidate_pool=200, n_val=40, n_test=40)
            row = {"benchmark": spec.name, "seed": seed, "n_train": result["n_train_final"]}
            row.update(result["final_metrics"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.groupby("benchmark").agg(["mean", "std"]))


if __name__ == "__main__":
    main()
