import torch
from torch import nn
from ltc import LTCCell


class ActorCritic(nn.Module):
    """Simple actor-critic network using an LTC cell."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 64, dt: float = 0.02):
        super().__init__()
        self.ltc = LTCCell(obs_size, hidden_size, dt)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    @property
    def hidden_size(self) -> int:
        return self.ltc.hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        h = self.ltc(x, h)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value.squeeze(-1), h
