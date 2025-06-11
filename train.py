import torch
import torch.nn.functional as F
import gymnasium as gym
from actor_critic import ActorCritic


def train(num_episodes: int = 300, gamma: float = 0.99):
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = ActorCritic(obs_size, action_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        h = torch.zeros(1, model.hidden_size)
        log_probs = []
        values = []
        rewards = []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, value, h = model(obs_t, h)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            values.append(value)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)

        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)
        advantage = returns - values.detach()

        policy_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}: reward={sum(rewards):.1f} loss={loss.item():.3f}")

    env.close()


if __name__ == "__main__":
    train()
