from activeem.acquisition import AcquisitionConfig
from activeem.active_loop import ActiveLearningRunner, TrainConfig
from activeem.benchmarks import get_benchmark_spec
from activeem.solver import SyntheticEMSolver


def test_active_loop_smoke():
    spec = get_benchmark_spec("filter")
    runner = ActiveLearningRunner(
        spec,
        SyntheticEMSolver(),
        TrainConfig(epochs=2, ensemble_size=1, mc_samples=1, batch_size=8),
        AcquisitionConfig(),
        seed=0,
        out_dir="runs/test_smoke",
    )
    result = runner.run(rounds=1, initial=8, batch_size=2, candidate_pool=20, n_val=8, n_test=8)
    assert result["n_train_final"] == 10
    assert "mae" in result["final_metrics"]
