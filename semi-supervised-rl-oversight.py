import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange

class SemiSupervisedEnv:
    def __init__(self, size=5, n_messes=5, max_steps=100, seed=None):
        # Kernel: initialize environment parameters and seed
        self.size = size
        self.n_messes = n_messes
        self.max_steps = max_steps
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()

    def reset(self):
        # : reset agent position, messes, steps, and cleaned count
        self.agent_pos = (0, 0)
        self.messes = set()
        while len(self.messes) < self.n_messes:
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if pos != self.agent_pos:
                self.messes.add(pos)
        self.cleaned_count = 0
        self.steps = 0
        return self.get_state()

    def get_state(self):
        # Kernel: return hashable state representation
        return (self.agent_pos[0], self.agent_pos[1], frozenset(self.messes))

    def step(self, action, true_reward_prob=0.15):
        # : apply action, update state, return observed and true reward
        self.steps += 1
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        self.agent_pos = (int(np.clip(self.agent_pos[0] + dx, 0, self.size - 1)),
                          int(np.clip(self.agent_pos[1] + dy, 0, self.size - 1)))
        cleaned_this_step = False
        true_reward = 0.0
        if self.agent_pos in self.messes:
            self.messes.remove(self.agent_pos)
            self.cleaned_count += 1
            true_reward = 1.0
            cleaned_this_step = True
        proxy_reward = 1.0 if cleaned_this_step else 0.0
        observed_reward = true_reward if random.random() < true_reward_prob else proxy_reward
        done = (len(self.messes) == 0) or (self.steps >= self.max_steps)
        return self.get_state(), observed_reward, true_reward, done


def train(n_episodes=8000, alpha=0.15, gamma=0.98, 
          epsilon_start=1.0, epsilon_end=0.02, epsilon_decay_steps=4000,
          true_reward_prob=0.15, seed=42):
    # : train agent using semi-supervised Q-learning
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    env = SemiSupervisedEnv(size=5, n_messes=6, max_steps=120, seed=seed)
    Q = defaultdict(float)
    obs_history, true_history = [], []
    epsilon = epsilon_start

    for episode in trange(n_episodes, desc="Training"):
        state = env.reset()
        done = False
        ep_obs_reward, ep_true_reward = 0.0, 0.0
        while not done:
            action = random.randint(0, 3) if random.random() < epsilon else np.argmax([Q[(state, a)] for a in range(4)])
            next_state, obs_r, true_r, done = env.step(action, true_reward_prob)
            best_next = max(Q[(next_state, a)] for a in range(4))
            Q[(state, action)] += alpha * ((true_r + gamma * best_next) - Q[(state, action)])
            state = next_state
            ep_obs_reward += obs_r
            ep_true_reward += true_r
        obs_history.append(ep_obs_reward)
        true_history.append(ep_true_reward)
        progress = min(1.0, episode / epsilon_decay_steps)
        epsilon = max(epsilon_end, epsilon_start + progress * (epsilon_end - epsilon_start))

    return obs_history, true_history


if __name__ == "__main__":
    # : train and visualize semi-supervised Q-learning results
    print("Training semi-supervised Q-learning (learning from TRUE reward when possible)...\n")
    obs_h, true_h = train(
        n_episodes=10000,
        alpha=0.12,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.015,
        epsilon_decay_steps=5000,
        true_reward_prob=0.12,
        seed=42
    )
    def smooth(x, w=200):
        # : simple moving average smoothing
        return np.convolve(x, np.ones(w)/w, mode='valid')
    plt.figure(figsize=(11, 5))
    plt.plot(smooth(true_h), label="True reward (cleaned messes)", color='#1f77b4', linewidth=2.3)
    plt.plot(smooth(obs_h), label="Observed (proxy) reward", color='#ff7f0e', linestyle='--', alpha=0.75, linewidth=1.8)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward per episode")
    plt.title("Semi-supervised Q-Learning\n(learning mostly from true reward when available)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
