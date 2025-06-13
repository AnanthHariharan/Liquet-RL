import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


class LTCCell(nn.Module):
    """Liquid Time‑Constant recurrent cell (robust)."""

    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.02):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_buffer("dt", torch.tensor(float(dt)))

        # Parameters
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        # Store time constants in log‑space; initialise τ ≈ 0.1 s (log τ ≈ −2.3)
        self.log_tau = nn.Parameter(torch.full((hidden_size,), -2.302585))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh, gain=0.1)
        # Reset bias and time constants to defaults
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.log_tau, -2.302585)  # ln(0.1)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)

        tau = F.softplus(self.log_tau) + 1e-3                        # Positive, smooth
        pre_act = torch.addmm(self.bias, x, self.weight_ih.t()) + h @ self.weight_hh.t()
        f = torch.tanh(pre_act)

        # Semi‑implicit Euler update for stability
        h = (h + self.dt * f) / (1.0 + self.dt / tau)
        return h

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, dt={self.dt.item():.4f})")
