import sys
import os
# Ensure project root is on path to find the `dreamer` package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import yaml, argparse, random, torch, gymnasium as gym
from pathlib import Path
from dreamer.model import DreamerModel
from dreamer.actor import Actor
from dreamer.critic import Critic, soft_update, hard_update, freeze
from dreamer.learner import DreamerLearner, symlog, symexp, lambda_return
from dreamer.replay import ReplayBuffer

# ------------------------------------------------------------------- #
# 1.  Parse flags and load config block(s)
# ------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+", default=["base", "cartpole"])
parser.add_argument("--logdir",  type=Path, default=Path("./logdir"))
args = parser.parse_args()

cfg = {}
for block in args.configs:
    cfg.update(yaml.safe_load(open("dreamer/configs.yaml"))[block])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0); torch.manual_seed(0)

# ------------------------------------------------------------------- #
# 2.  Environment and replay buffer
# ------------------------------------------------------------------- #
env   = gym.make(cfg["env_id"])
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if cfg["action_discrete"] else env.action_space.shape[0]

replay = ReplayBuffer(capacity=100_000,
                      obs_shape=(obs_dim,),
                      act_shape=(act_dim,),
                      device=device)

# ------------------------------------------------------------------- #
# 3.  Build world-model, actor, critic, learner
# ------------------------------------------------------------------- #
wm     = DreamerModel(obs_dim, act_dim,
                      deter_dim=cfg["deter_dim"],
                      stoch_dim=cfg["stoch_dim"])

actor  = Actor(wm.rssm.deter_dim + wm.rssm.stoch_dim,
               action_shape=(act_dim,),
               discrete=cfg["action_discrete"])

critic = Critic(wm.rssm.deter_dim + wm.rssm.stoch_dim)

learner = DreamerLearner(
    wm,
    actor,
    critic,
    lr_model=float(cfg["lr_model"]),
    lr_actor=float(cfg["lr_actor"]),
    lr_critic=float(cfg["lr_critic"]),
    imag_horizon=int(cfg["imag_horizon"]),
    discount=float(cfg["discount"]),
    lambda_=float(cfg["lambda_"]),
    tau_target=float(cfg["tau_target"]),
    kl_beta=float(cfg["kl_beta"]),
    free_nats=float(cfg["free_nats"]),
    device=device,
)

# ------------------------------------------------------------------- #
# 4.  Training loop
# ------------------------------------------------------------------- #
obs, _ = env.reset()
h, z = wm.init_state(batch_size=1, device=device)
prev_act = torch.zeros(1, act_dim, device=device)
episode_return = 0; step = 0
while step < 1_000_000:
    # ----- collect real step ----- #
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    h, z, feat = wm.step(obs_t, prev_act, h, z)
    action, _, _ = actor(feat, deterministic=False)
    if cfg["action_discrete"]:
        act_idx = int(action.item())
        act_np = act_idx
        prev_act = torch.zeros(1, act_dim, device=device)
        prev_act[0, act_idx] = 1.0
    else:
        act_np = action.squeeze(0).cpu().numpy()
        prev_act = action.detach()

    next_obs, reward, done, truncated, _ = env.step(act_np)
    replay.add(obs, prev_act.squeeze(0).cpu().numpy(), reward, done)
    obs, episode_return = (next_obs, episode_return + reward)

    if done or truncated:
        print(f"Episode return {episode_return:.1f}")
        obs, _ = env.reset(); episode_return = 0
        h, z = wm.init_state(batch_size=1, device=device)
        prev_act.zero_()

    # ----- learner updates ----- #
    if replay.ready(cfg["batch_size"], cfg["seq_len"]):
        for _ in range(cfg["train_ratio"]):
            batch = replay.sample(cfg["batch_size"], cfg["seq_len"])
            metrics = learner.step(batch)
        if step % cfg["log_every"] == 0:
            print({k:f"{v:.3f}" for k,v in metrics.items()})

    step += 1
