"""Simple circular replay buffer that returns contiguous **time‑major**
sequences suitable for Dreamer‑style training.

The buffer stores (obs, act, rew, done) tuples. `sample()` returns a dict:

    {
        "obs":  Tensor[seq_len, batch, *obs_shape],
        "act":  Tensor[seq_len, batch, *act_shape],
        "rew":  Tensor[seq_len, batch, 1],
        "done": Tensor[seq_len, batch, 1],
    }

All tensors live on the device specified at construction time.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        device: torch.device | str = "cpu",
        dtype_obs: torch.dtype = torch.float32,
        dtype_act: torch.dtype = torch.float32,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)

        self.obs = torch.zeros((capacity, *obs_shape), dtype=dtype_obs, device=self.device)
        self.act = torch.zeros((capacity, *act_shape), dtype=dtype_act, device=self.device)
        self.rew = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

        self.idx: int = 0
        self.full: bool = False

    # ------------------------------------------------------------------ #
    #                            Book‑keeping                            #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def ready(self, batch_size: int, seq_len: int) -> bool:
        """Return *True* when enough transitions are stored for sampling."""
        buf_len = len(self)
        return buf_len > seq_len and buf_len >= batch_size

    # ------------------------------------------------------------------ #
    #                               Add                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def add(self, obs, act, rew: float, done: float):
        """Store a single transition (obs_t, act_t, rew_t, done_t)."""
        self.obs[self.idx].copy_(torch.as_tensor(obs, device=self.device))
        self.act[self.idx].copy_(torch.as_tensor(act, device=self.device))
        self.rew[self.idx] = rew
        self.done[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    # ------------------------------------------------------------------ #
    #                              Sample                                #
    # ------------------------------------------------------------------ #
    def _valid_start(self, seq_len: int) -> int:
        """Return a random start index where the sequence of length `seq_len`
        does not cross the circular buffer write‑pointer and contains no done=1
        internally (episode boundary)."""
        max_idx = self.capacity if self.full else self.idx
        while True:
            start = random.randint(0, max_idx - seq_len - 1)
            end = (start + seq_len) % self.capacity

            # sequence must not wrap over the current write index
            if start < self.idx <= end and not self.full:
                continue
            # ensure no terminal flags inside the sequence (except at final step)
            if self.done[start : start + seq_len - 1].any():
                continue
            return start

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int):
        """Return a batch of contiguous sequences (time‑major)."""
        idxs = [self._valid_start(seq_len) for _ in range(batch_size)]

        obs_batch = torch.stack(
            [self.obs[i : i + seq_len] for i in idxs], dim=1
        )  # (seq, batch, *obs)

        act_batch = torch.stack(
            [self.act[i : i + seq_len] for i in idxs], dim=1
        )  # (seq, batch, *act)

        rew_batch = torch.stack(
            [self.rew[i : i + seq_len] for i in idxs], dim=1
        )  # (seq, batch, 1)

        done_batch = torch.stack(
            [self.done[i : i + seq_len] for i in idxs], dim=1
        )  # (seq, batch, 1)

        return {
            "obs": obs_batch,
            "act": act_batch,
            "rew": rew_batch,
            "done": done_batch,
        }
