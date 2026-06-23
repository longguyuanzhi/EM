from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def data_loss(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], scalar_weight: float = 0.2) -> torch.Tensor:
    return F.l1_loss(pred["spectra"], batch["spectra"]) + scalar_weight * F.l1_loss(pred["scalars"], batch["scalars"])


def passivity_loss(spectra: torch.Tensor) -> torch.Tensor:
    """Magnitude-only passivity proxy; replace with complex S-matrix norm if available."""
    energy = torch.sqrt(torch.sum(spectra ** 2, dim=1) + 1e-8)
    return torch.mean(torch.relu(energy - 1.0) ** 2)


def reciprocity_loss(spectra: torch.Tensor, port_pairs: Optional[list[tuple[int, int]]] = None) -> torch.Tensor:
    if not port_pairs:
        return spectra.new_tensor(0.0)
    terms = []
    for i, j in port_pairs:
        if i < spectra.shape[1] and j < spectra.shape[1]:
            terms.append(F.mse_loss(spectra[:, i, :], spectra[:, j, :]))
    return torch.stack(terms).mean() if terms else spectra.new_tensor(0.0)


def smoothness_loss(spectra: torch.Tensor) -> torch.Tensor:
    if spectra.shape[-1] < 3:
        return spectra.new_tensor(0.0)
    d2 = spectra[:, :, 2:] - 2 * spectra[:, :, 1:-1] + spectra[:, :, :-2]
    return torch.mean(torch.abs(d2))


def scalar_spectrum_consistency_loss(pred: Dict[str, torch.Tensor], benchmark: str, freq: torch.Tensor) -> torch.Tensor:
    spectra, scalars = pred["spectra"], pred["scalars"]
    if scalars.numel() == 0:
        return spectra.new_tensor(0.0)
    if benchmark == "filter" and spectra.shape[1] >= 2:
        w = torch.softmax(25.0 * spectra[:, 1, :], dim=-1)
        f0_hat = (w * freq.view(1, -1)).sum(dim=-1)
        return F.l1_loss(scalars[:, 0], f0_hat)
    if benchmark == "resonator":
        w = torch.softmax(-25.0 * spectra[:, 0, :], dim=-1)
        f0_hat = (w * freq.view(1, -1)).sum(dim=-1)
        return F.l1_loss(scalars[:, 0], f0_hat)
    if benchmark == "coupler" and spectra.shape[1] >= 3:
        coupling = 20.0 * torch.log10(torch.clamp(torch.max(spectra[:, 2, :], dim=-1).values, min=1e-4))
        return F.l1_loss(scalars[:, 0], coupling)
    return spectra.new_tensor(0.0)


def physics_regularization(pred: Dict[str, torch.Tensor], benchmark: str, freq: torch.Tensor, weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """Response-domain physics regularization, not full-field Maxwell residual."""
    weights = weights or {}
    return (
        float(weights.get("passivity", 0.20)) * passivity_loss(pred["spectra"])
        + float(weights.get("reciprocity", 0.10)) * reciprocity_loss(pred["spectra"], [])
        + float(weights.get("smoothness", 0.05)) * smoothness_loss(pred["spectra"])
        + float(weights.get("consistency", 0.20)) * scalar_spectrum_consistency_loss(pred, benchmark, freq)
    )


def total_loss(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], benchmark: str, lambda_phy: float = 0.1, lambda_reg: float = 0.0, model: Optional[torch.nn.Module] = None) -> torch.Tensor:
    loss = data_loss(pred, batch)
    if lambda_phy > 0:
        loss = loss + lambda_phy * physics_regularization(pred, benchmark, batch["freq"])
    if lambda_reg > 0 and model is not None:
        reg = sum((p ** 2).sum() for p in model.parameters())
        loss = loss + lambda_reg * reg
    return loss
