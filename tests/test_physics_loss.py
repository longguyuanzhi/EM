import torch

from activeem.losses import physics_regularization


def test_physics_regularization_finite():
    pred = {"spectra": torch.rand(4, 2, 31), "scalars": torch.rand(4, 3)}
    freq = torch.linspace(1.8, 3.2, 31)
    loss = physics_regularization(pred, "filter", freq)
    assert torch.isfinite(loss)
