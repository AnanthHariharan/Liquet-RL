"""Value function for the Dreamer‑LTC agent.

The critic maps latent features (concatenation of deterministic hidden state `h`
and stochastic state `z`) to a state‑value estimate *V(s)*.  A separate target
critic is maintained for TD targets and updated by soft EMA.

Usage:
    critic   = Critic(feat_dim)
    target   = Critic(feat_dim).load_state_dict(critic.state_dict())
    ...
    value    = critic(feat)                 # forward pass
    soft_update(critic, target, tau=0.01)   # after each optimisation step
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, inp: int, out: int = 1, hid: int = 512, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(inp, hid), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU()]
        layers.append(nn.Linear(hid, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """Dreamer critic head that outputs scalar state value."""

    def __init__(self, feat_dim: int, hid_dim: int = 512, depth: int = 3):
        super().__init__()
        self.net = MLP(feat_dim, 1, hid_dim, depth)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns V(s) as shape (batch,)."""
        return self.net(feat).squeeze(-1)


# --------------------------------------------------------------------------- #
#                   Target‑net utilities (soft / hard copy)                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float = 0.01) -> None:
    """EMA update of target network parameters."""
    for p_src, p_tgt in zip(source.parameters(), target.parameters(), strict=True):
        p_tgt.data.lerp_(p_src.data, tau)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Hard copy parameters from source to target."""
    target.load_state_dict(copy.deepcopy(source.state_dict()))


def freeze(module: nn.Module) -> None:
    """Disable gradient computation for the given module."""
    for p in module.parameters():
        p.requires_grad_(False)
