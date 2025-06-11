

"""Stochastic actor for the Dreamer‑LTC agent.

Supports both discrete (categorical) and continuous (tanh‑squashed Gaussian)
action spaces. The actor maps latent features produced by the world‑model
(typically `torch.cat([h, z], -1)`) to a distribution over actions.
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal, TransformedDistribution, TanhTransform


class MLP(nn.Module):
    """Small feed‑forward network with configurable depth."""

    def __init__(self, inp_dim: int, out_dim: int, hid_dim: int = 256, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(inp_dim, hid_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid_dim, hid_dim), nn.SiLU()]
        layers.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    """Dreamer actor head.

    Args:
        feat_dim:  Dimension of the latent feature input.
        action_shape: Tuple describing the action space shape e.g. ``(4,)``.
        discrete: Whether the action space is discrete.
        min_std:  Lower bound for the std dev when using a Gaussian policy.
    """

    def __init__(
        self,
        feat_dim: int,
        action_shape: Tuple[int, ...],
        *,
        discrete: bool,
        min_std: float = 1e-4,
        hid_dim: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        self.discrete = discrete
        self.act_size = int(torch.tensor(action_shape).prod())
        self.min_std = min_std

        out_dim = self.act_size * 2 if not discrete else self.act_size
        self.backbone = MLP(feat_dim, out_dim, hid_dim, depth)

    # --------------------------------------------------------------------- #
    # Distribution helpers
    # --------------------------------------------------------------------- #
    def _dist_cont(self, logits: torch.Tensor) -> TransformedDistribution:
        """Gaussian with learned μ, σ then tanh‑squashed to (−1, 1)."""
        mu, log_std = logits.chunk(2, -1)
        std = torch.exp(log_std.clamp(-10, 2)) + self.min_std
        base = Normal(mu, std)
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])

    def _dist_disc(self, logits: torch.Tensor) -> Categorical:
        return Categorical(logits=logits)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def dist(self, feat: torch.Tensor):
        logits = self.backbone(feat)
        return self._dist_disc(logits) if self.discrete else self._dist_cont(logits)

    def forward(self, feat: torch.Tensor, *, deterministic: bool = False):
        """Returns action, log‑prob, and distribution."""
        dist = self.dist(feat)
        if deterministic:
            action = dist.mean if hasattr(dist, "mean") else dist.probs.argmax(-1)
        else:
            action = dist.rsample() if hasattr(dist, "rsample") else dist.sample()

        log_prob = dist.log_prob(action)
        # Ensure correct shape for multi‑dimensional continuous actions
        if not self.discrete and action.ndim > log_prob.ndim:
            log_prob = log_prob.sum(-1)

        return action, log_prob, dist.entropy()

    # --------------------------------------------------------------------- #
    # Convenience for behaviour‑cloning loss or offline RL
    # --------------------------------------------------------------------- #
    def log_prob(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.dist(feat)
        lp = dist.log_prob(action)
        return lp if self.discrete else lp.sum(-1)