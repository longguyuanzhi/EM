from __future__ import annotations

from activeem.acquisition import AcquisitionConfig
from activeem.active_loop import ActiveLearningRunner, TrainConfig
from activeem.benchmarks import get_benchmark_spec
from activeem.solver import SyntheticEMSolver


def main() -> None:
    spec = get_benchmark_spec("filter")
    runner = ActiveLearningRunner(
        spec=spec,
        solver=SyntheticEMSolver(noise_std=0.0),
        train_cfg=TrainConfig(epochs=20, batch_size=16, ensemble_size=2, mc_samples=2, device="cpu"),
        acq_cfg=AcquisitionConfig(),
        seed=0,
        out_dir="runs/quickstart_filter",
    )
    result = runner.run(rounds=1, initial=16, batch_size=4, candidate_pool=120, n_val=32, n_test=32)
    print("Final metrics:")
    for k, v in result["final_metrics"].items():
        print(f"  {k}: {v:.5f}")


if __name__ == "__main__":
    main()
