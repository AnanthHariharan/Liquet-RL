

"""Dreamer-LTC learner.

Handles:
    • World‑model update   (reconstruction + reward + KL)
    • Imagined rollout     (latent RSSM imagine)
    • Actor / critic update (λ‑return)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions.kl import kl_divergence

from .critic import Critic, soft_update
from .actor import Actor
from .model import DreamerModel


# --------------------------------------------------------------------------- #
#                       Numeric helpers (symlog, λ‑return)                   #
# --------------------------------------------------------------------------- #
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute λ‑return (time‑major tensors)."""
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], 0)
    target = next_values * (1.0 - lambda_) + values * lambda_
    returns = []
    ret = bootstrap
    for t in reversed(range(rewards.size(0))):
        ret = rewards[t] + discounts[t] * ret
        returns.append(ret)
        ret = target[t] + discounts[t] * lambda_ * (ret - target[t])
    returns = torch.stack(list(reversed(returns)))
    return returns


# --------------------------------------------------------------------------- #
#                                   Learner                                   #
# --------------------------------------------------------------------------- #
class DreamerLearner:
    def __init__(
        self,
        model: DreamerModel,
        actor: Actor,
        critic: Critic,
        *,
        lr_model: float = 6e-4,
        lr_actor: float = 8e-5,
        lr_critic: float = 8e-5,
        imag_horizon: int = 15,
        discount: float = 0.99,
        lambda_: float = 0.95,
        kl_beta: float = 0.1,
        free_nats: float = 1.5,
        tau_target: float = 0.01,
        device: torch.device | str = "cpu",
    ):
        self.model = model.to(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.target_critic = Critic(
            self.critic.net.net[0].in_features
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.opt_model = optim.Adam(self.model.parameters(), lr=lr_model)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.imag_horizon = imag_horizon
        self.discount = discount
        self.lambda_ = lambda_
        self.kl_beta = kl_beta
        self.free_nats = free_nats
        self.tau_target = tau_target
        self.device = device

    # --------------------------------------------------------------------- #
    # Main training step
    # --------------------------------------------------------------------- #
    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        batch keys: 'obs'  (T,B,D), 'act' (T,B,A), 'rew' (T,B,1), 'done' (T,B,1)
        All tensors are assumed to be torch.float32 except 'done' (float 0/1).
        """
        obs = batch["obs"].to(self.device)          # (T,B,obs_dim)
        act = batch["act"].to(self.device)          # (T,B,A)
        rew = batch["rew"].to(self.device)          # (T,B,1)
        done = batch["done"].to(self.device)        # (T,B,1)
        disc = (1.0 - done) * self.discount

        # ---------------- World‑model update ---------------- #
        priors, posts, obs_pred, rew_pred, feat = self.model(obs, act)
        # reconstruction
        obs_loss = F.mse_loss(obs_pred, obs, reduction="mean")
        rew_loss = F.mse_loss(rew_pred.squeeze(-1), rew.squeeze(-1), reduction="mean")

        # KL divergence time‑major list → tensor
        kl = torch.stack([kl_divergence(q, p) for p, q in zip(priors, posts)])
        kl = torch.mean(torch.clamp(kl, min=self.free_nats))
        world_loss = obs_loss + rew_loss + self.kl_beta * kl

        self.opt_model.zero_grad()
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        self.opt_model.step()

        # ---------------- Imagination rollout ---------------- #
        with torch.no_grad():
            last_z = torch.stack([post.rsample() for post in posts])[-1]  # (B,Z)
        h_img, z_img = self.model.rssm.imagine(
            last_z, act[1 : self.imag_horizon + 1], self.imag_horizon
        )
        feat_img = torch.cat([h_img, z_img], -1)  # (H,B,feat)
        rew_img = self.model.reward_decoder(feat_img).squeeze(-1)
        disc_img = torch.full_like(rew_img, self.discount)

        # value targets
        value_bootstrap = self.target_critic(feat_img[-1])
        value_img = self.target_critic(feat_img[:-1])
        ret_img = lambda_return(
            rew_img[:-1], value_img, disc_img[:-1], value_bootstrap, self.lambda_
        )

        # ---------------- Critic update ---------------- #
        value_pred = self.critic(feat_img[:-1].detach())
        critic_loss = F.mse_loss(value_pred, ret_img.detach())

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.opt_critic.step()

        # ---------------- Actor update ---------------- #
        act_dist = self.actor.dist(feat_img[:-1].detach())
        action = act_dist.rsample()
        log_prob = act_dist.log_prob(action)
        if log_prob.ndim > 2:  # continuous multi‑dim
            log_prob = log_prob.sum(-1)
        advantage = (ret_img.detach() - value_pred.detach())
        actor_loss = -(log_prob * advantage).mean() - 0.01 * act_dist.entropy().mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.opt_actor.step()

        # ---------------- Target critic EMA ---------------- #
        soft_update(self.critic, self.target_critic, self.tau_target)

        return {
            "world_loss": world_loss.item(),
            "obs_loss": obs_loss.item(),
            "rew_loss": rew_loss.item(),
            "kl": kl.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }