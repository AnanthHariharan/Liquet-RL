import torch
from torch import nn
from torch.distributions.normal import Normal
from ltc import LTCCell


class MLP(nn.Module):
    def __init__(self, inp, out, hid=256, depth=2):
        super().__init__()
        layers = [nn.Linear(inp, hid), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU()]
        layers += [nn.Linear(hid, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RSSM(nn.Module):
    def __init__(self, obs_dim, act_dim, deter_dim=256, stoch_dim=32, dt=0.02):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        inp_dim = stoch_dim + act_dim

        self.ltc = LTCCell(inp_dim, deter_dim, dt)
        self.prior = MLP(deter_dim, 2 * stoch_dim)
        self.post = MLP(deter_dim + obs_dim, 2 * stoch_dim)

    def _dist(self, stats):
        mean, log_std = stats.chunk(2, -1)
        std = torch.exp(log_std.clamp(-10, 1))
        return Normal(mean, std)

    def observe(self, obs, act):
        T = obs.size(0)
        h = torch.zeros(obs.size(1), self.deter_dim, device=obs.device)
        prior_dists, post_dists, stoch_states, deter_states = [], [], [], []

        for t in range(T):
            x = torch.cat([
                act[t - 1] if t else torch.zeros_like(act[0]),
                stoch_states[-1] if t else torch.zeros_like(torch.zeros(obs.size(1), self.stoch_dim, device=obs.device))
            ], dim=-1)
            h = self.ltc(x, h)

            prior = self._dist(self.prior(h))
            deter_states.append(h)
            prior_dists.append(prior)

            inp = torch.cat([h, obs[t]], -1)
            post = self._dist(self.post(inp))
            z = post.rsample()
            post_dists.append(post)
            stoch_states.append(z)

        return prior_dists, post_dists, torch.stack(deter_states), torch.stack(stoch_states)

    def imagine(self, z0, act, horizon):
        h = torch.zeros(z0.size(0), self.deter_dim, device=z0.device)
        zs, hs = [z0], []
        for t in range(horizon):
            x = torch.cat([zs[-1], act[t]], -1)
            h = self.ltc(x, h)
            stats = self.prior(h)
            z = self._dist(stats).sample()
            zs.append(z)
            hs.append(h)
        return torch.stack(hs), torch.stack(zs[1:])


class DreamerModel(nn.Module):
    def __init__(self, obs_dim, act_dim, deter_dim=256, stoch_dim=32):
        super().__init__()
        self.encoder = MLP(obs_dim, obs_dim * 2)
        self.rssm = RSSM(obs_dim * 2, act_dim, deter_dim, stoch_dim)
        self.obs_decoder = MLP(deter_dim + stoch_dim, obs_dim)
        self.reward_decoder = MLP(deter_dim + stoch_dim, 1)

    def forward(self, obs_seq, act_seq):
        obs_emb = self.encoder(obs_seq)
        priors, posts, hs, zs = self.rssm.observe(obs_emb, act_seq)
        feat = torch.cat([hs, zs], -1)
        obs_pred = self.obs_decoder(feat)
        rew_pred = self.reward_decoder(feat)
        return priors, posts, obs_pred, rew_pred, feat

    def init_state(self, batch_size: int = 1, device: torch.device | None = None):
        """Return zero-initialised deterministic and stochastic states."""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.rssm.deter_dim, device=device)
        # Sample initial stochastic state from the prior distribution
        prior_stats = self.rssm.prior(h)
        prior_dist = self.rssm._dist(prior_stats)
        z = prior_dist.rsample()
        return h, z

    def step(self, obs: torch.Tensor, act: torch.Tensor,
             h: torch.Tensor, z: torch.Tensor):
        """
        One RSSM update:
          1) Encode observation
          2) Update deterministic hidden via LTC ODE
          3) Compute prior from new hidden state
          4) Compute posterior incorporating observation
          5) Sample new stochastic state
          6) Return (h_next, z_next, feature)
        """
        # 1) embed the raw observation
        obs_emb = self.encoder(obs)  # (B, obs_dim*2)

        # 2) deterministic update via LTC cell
        x = torch.cat([z, act], dim=-1)    # (B, stoch_dim + act_dim)
        h_next = self.rssm.ltc(x, h)       # (B, deter_dim)

        # 3) prior distribution from h_next
        prior_stats = self.rssm.prior(h_next)   # (B, 2*stoch_dim)
        prior_dist = self.rssm._dist(prior_stats)

        # 4) posterior distribution from h_next and obs_emb
        post_input = torch.cat([h_next, obs_emb], dim=-1)
        post_stats = self.rssm.post(post_input) # (B, 2*stoch_dim)
        post_dist = self.rssm._dist(post_stats)

        # 5) sample new stochastic state
        z_next = post_dist.rsample()           # (B, stoch_dim)

        # 6) form feature for actor/critic
        feat = torch.cat([h_next, z_next], dim=-1)  # (B, deter_dim + stoch_dim)

        return h_next, z_next, feat

    def step_feature(self, obs: torch.Tensor, act: torch.Tensor, h: torch.Tensor, z: torch.Tensor):
        """
        Single-step update that returns the combined feature for actor/critic.
        Uses LTC for deterministic update and posterior for stochastic state.
        """
        h_next, z_next, feat = self.step(obs, act, h, z)
        return h_next, z_next, feat
