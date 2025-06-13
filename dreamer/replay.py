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
        """Return True when the buffer holds enough data for sampling sequences of length seq_len."""
        # Need at least batch_size entries for batches and seq_len for a full sequence
        return len(self) >= batch_size and len(self) >= seq_len

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
        max_idx = self.capacity if self.full else self.idx
        upper = max_idx - seq_len
        if upper < 0:
            raise ValueError(f"Not enough entries ({len(self)}) to sample a sequence of length {seq_len}")
        return random.randint(0, upper)

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int):
        """
        Return a batch of contiguous sequences (time-major).
        Handles wrap-around sequences.
        """
        idxs = [self._valid_start(seq_len) for _ in range(batch_size)]

        def grab(buf, start):
            # Handles wrap-around
            end = start + seq_len
            if end <= self.capacity:
                return buf[start:end]
            else:
                first = buf[start:self.capacity]
                second = buf[0:end - self.capacity]
                return torch.cat([first, second], dim=0)

        obs_batch = torch.stack([grab(self.obs, i) for i in idxs], dim=1)
        act_batch = torch.stack([grab(self.act, i) for i in idxs], dim=1)
        rew_batch = torch.stack([grab(self.rew, i) for i in idxs], dim=1)
        done_batch = torch.stack([grab(self.done, i) for i in idxs], dim=1)

        return {
            "obs": obs_batch,
            "act": act_batch,
            "rew": rew_batch,
            "done": done_batch,
        }
