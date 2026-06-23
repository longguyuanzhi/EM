from __future__ import annotations

import torch
from torch import nn


def canonicalize_symmetric_design(x: torch.Tensor, benchmark: str) -> torch.Tensor:
    """Canonicalize repeated/mirrored variables for known layouts."""
    y = x.clone()
    if benchmark == "filter" and y.shape[-1] >= 2:
        y[..., 0:2] = torch.sort(y[..., 0:2], dim=-1).values
    if benchmark == "coupler" and y.shape[-1] >= 5:
        y[..., 0:2] = torch.sort(y[..., 0:2], dim=-1).values
        y[..., 3:5] = torch.sort(y[..., 3:5], dim=-1).values
    return y


class FrequencyEncoding(nn.Module):
    """Sin/cos frequency features plus normalized frequency."""

    def __init__(self, n_harmonics: int = 4):
        super().__init__()
        self.n_harmonics = int(n_harmonics)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_harmonics + 1

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        if freq.dim() == 1:
            f = freq.unsqueeze(0)
        else:
            f = freq
        f_norm = (f - f.amin(dim=-1, keepdim=True)) / (f.amax(dim=-1, keepdim=True) - f.amin(dim=-1, keepdim=True) + 1e-8)
        feats = [f_norm]
        for h in range(1, self.n_harmonics + 1):
            feats.append(torch.sin(2 * torch.pi * h * f_norm))
            feats.append(torch.cos(2 * torch.pi * h * f_norm))
        return torch.stack(feats, dim=-1)


def build_frequency_conditioned_input(x: torch.Tensor, freq: torch.Tensor, encoder: FrequencyEncoding, benchmark: str) -> torch.Tensor:
    """Build [B,F,D+Fenc] sequence z0(j)=P([psi(x),phi(f_j)])."""
    x = canonicalize_symmetric_design(x, benchmark)
    if freq.dim() == 1:
        freq_batch = freq.unsqueeze(0).expand(x.shape[0], -1)
    else:
        freq_batch = freq
    fenc = encoder(freq_batch)
    x_rep = x.unsqueeze(1).expand(-1, freq_batch.shape[1], -1)
    return torch.cat([x_rep, fenc], dim=-1)
