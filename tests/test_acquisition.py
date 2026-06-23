import numpy as np

from activeem.acquisition import AcquisitionConfig, select_batch
from activeem.uncertainty import PredictionStats


def test_select_batch_shapes():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(-1, 1, size=(n, 5)).astype("float32")
    stats = PredictionStats(
        mean_spectra=rng.random((n, 2, 21)).astype("float32"),
        mean_scalars=rng.random((n, 3)).astype("float32"),
        std_spectra=0.05 + 0.01 * rng.random((n, 2, 21)).astype("float32"),
        std_scalars=0.05 + 0.01 * rng.random((n, 3)).astype("float32"),
    )
    out = select_batch(x, stats, x[:10], 5, AcquisitionConfig())
    assert out["x"].shape == (5, 5)
    assert len(out["indices"]) == 5
