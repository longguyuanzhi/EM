from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .encoding import FrequencyEncoding, build_frequency_conditioned_input


class SpectralConv1d(nn.Module):
    """1D Fourier convolution over the frequency axis."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        scale = 1.0 / max(1, width * width)
        self.weight = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,F,C]
        b, f, c = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = torch.zeros(b, x_ft.shape[1], c, device=x.device, dtype=torch.cfloat)
        m = min(self.modes, x_ft.shape[1])
        out_ft[:, :m, :] = torch.einsum("bfi,ijm->bfj", x_ft[:, :m, :], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=f, dim=1)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int, dropout: float = 0.05):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.pointwise = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        return self.dropout(self.act(self.norm(y)))


class FrequencyConditionedFNO(nn.Module):
    """Frequency-conditioned FNO surrogate.

    Inputs:
        x: [B,D] normalized design vector.
        freq: [F] frequency grid.

    Outputs:
        spectra: [B,P,F].
        scalars: [B,S].
    """

    def __init__(
        self,
        design_dim: int,
        n_ports: int,
        n_scalars: int,
        benchmark: str,
        n_freq: int,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        dropout: float = 0.05,
        n_freq_harmonics: int = 4,
    ):
        super().__init__()
        self.design_dim = int(design_dim)
        self.n_ports = int(n_ports)
        self.n_scalars = int(n_scalars)
        self.benchmark = benchmark
        self.freq_encoder = FrequencyEncoding(n_freq_harmonics)
        self.lift = nn.Sequential(
            nn.Linear(self.design_dim + self.freq_encoder.out_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes, dropout) for _ in range(n_layers)])
        self.spectral_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_ports), nn.Sigmoid())
        self.scalar_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_scalars))

    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = build_frequency_conditioned_input(x, freq, self.freq_encoder, self.benchmark)
        h = self.lift(z)
        for block in self.blocks:
            h = h + block(h)
        spectra = self.spectral_head(h).transpose(1, 2).contiguous()
        scalars = self.scalar_head(h.mean(dim=1))
        return {"spectra": spectra, "scalars": scalars}


class MLPBaseline(nn.Module):
    """Plain vector-to-vector baseline."""

    def __init__(self, design_dim: int, n_ports: int, n_freq: int, n_scalars: int, width: int = 128, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        layers = []
        d = design_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
            d = width
        self.backbone = nn.Sequential(*layers)
        self.n_ports = n_ports
        self.n_freq = n_freq
        self.spectral_head = nn.Sequential(nn.Linear(width, n_ports * n_freq), nn.Sigmoid())
        self.scalar_head = nn.Linear(width, n_scalars)

    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "spectra": self.spectral_head(h).view(x.shape[0], self.n_ports, self.n_freq),
            "scalars": self.scalar_head(h),
        }
