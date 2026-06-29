import numpy as np
import matplotlib.pyplot as plt #type: ignore

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
WALLS = {
    (1, 1), (1, 2), (1, 3),
    (3, 0), (3, 1),
    (3, 5), (3, 6), (3, 7),
    (5, 2), (5, 3), (5, 4),
    (5, 7), (5, 8),
    (7, 1), (7, 2),
    (7, 5), (7, 6), (7, 7),
    (2, 8), (6, 4),
}

HOLES = {
    (2, 1), (4, 2), (4, 7), (6, 1), (6, 6),
    (8, 3), (1, 6), (8, 8), (3, 4), (9, 5),
}

START = (0, 0)
GOAL  = (9, 0)

ACTIONS = {
    0: ( 0,  1),   # UP
    1: ( 0, -1),   # DOWN
    2: (-1,  0),   # LEFT
    3: ( 1,  0),   # RIGHT
}

ORTHOGONAL = {
    0: (2, 3),
    1: (3, 2),
    2: (1, 0),
    3: (0, 1),
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MazeEnv:
    GRID_SIZE   = 10
    NUM_ACTIONS = 4
    NUM_STATES  = 100

    def __init__(self):
        self.rng   = np.random.default_rng()
        self.walls = WALLS
        self.holes = HOLES
        self.start = START
        self.goal  = GOAL
        self._state = (0, 0)

    def reset(self):
        self._state = self.start
        return self._state

    def step(self, action):
        p = self.rng.random()
        if p < 0.50:
            effective = action
        elif p < 0.75:
            effective = ORTHOGONAL[action][0]
        else:
            effective = ORTHOGONAL[action][1]

        dx, dy = ACTIONS[effective]
        x, y   = self._state
        nx, ny = x + dx, y + dy

        if (0 <= nx < self.GRID_SIZE and
                0 <= ny < self.GRID_SIZE and
                (nx, ny) not in self.walls):
            self._state = (nx, ny)

        if self._state == self.goal:
            reward, done = 1.0, True
        elif self._state in self.holes:
            reward, done = -0.5, False
            self._state  = self.start
        else:
            reward, done = -0.001, False

        return self._state, reward, done


# ---------------------------------------------------------------------------
# Agent — Decaying Epsilon Greedy (decays per episode, not per step)
# ---------------------------------------------------------------------------
def _idx(state):
    x, y = state
    return y * 10 + x


class DecayingEpsilonGreedy:
    def __init__(self, epsilon=1.0, decay=0.99):
        self.counts        = np.zeros((100, 4))
        self.values        = np.zeros((100, 4))
        self.epsilon       = epsilon
        self.decay         = decay
        self._init_epsilon = epsilon

    def pick_action(self, state):
        if np.random.random() >= self.epsilon:
            idx = _idx(state)
            action = int(np.argmax(self.values[idx]))
        else:
            action = np.random.randint(4)
        return action

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon *= self.decay

    def reset(self):
        self.counts  = np.zeros((100, 4))
        self.values  = np.zeros((100, 4))
        self.epsilon = self._init_epsilon


# ---------------------------------------------------------------------------
# Evaluation Strategies
# ---------------------------------------------------------------------------
MAX_STEPS = 2000


def sample_average(agent, env):
    done = False
    total_reward = 0
    steps = 0
    while not done and steps < MAX_STEPS:
        state  = env._state
        idx    = _idx(state)
        action = agent.pick_action(state)
        agent.counts[idx][action] += 1
        _, reward, done = env.step(action)
        agent.values[idx][action] += (reward - agent.values[idx][action]) / agent.counts[idx][action]
        total_reward += reward
        steps += 1
    return total_reward


def fv_monte_carlo(agent, env, gamma=0.99):
    episode = []
    visited = set()
    done    = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        state  = env._state
        action = agent.pick_action(state)
        _, reward, done = env.step(action)
        episode.append((state, action, reward))
        total_reward += reward
        steps += 1

    rewards = [r for _, _, r in episode]
    full_returns = [
        sum(gamma**j * rewards[i+j] for j in range(len(rewards)-i))
        for i in range(len(rewards))
    ]

    for i, (state, action, _) in enumerate(episode):
        if state not in visited:
            visited.add(state)
            idx = _idx(state)
            agent.counts[idx][action] += 1
            agent.values[idx][action] += (
                full_returns[i] - agent.values[idx][action]
            ) / agent.counts[idx][action]

    return total_reward


def ev_monte_carlo(agent, env, gamma=0.99):
    episode = []
    done    = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        state  = env._state
        action = agent.pick_action(state)
        _, reward, done = env.step(action)
        episode.append((state, action, reward))
        total_reward += reward
        steps += 1

    rewards = [r for _, _, r in episode]
    full_returns = [
        sum(gamma**j * rewards[i+j] for j in range(len(rewards)-i))
        for i in range(len(rewards))
    ]

    for i, (state, action, _) in enumerate(episode):
        idx = _idx(state)
        agent.counts[idx][action] += 1
        agent.values[idx][action] += (
            full_returns[i] - agent.values[idx][action]
        ) / agent.counts[idx][action]

    return total_reward


def temp_diff(agent, env, gamma=0.99):
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        state  = env._state
        idx    = _idx(state)
        action = agent.pick_action(state)
        agent.counts[idx][action] += 1
        new_state, reward, done = env.step(action)
        new_idx = _idx(new_state)

        future = 0 if done else gamma * np.max(agent.values[new_idx])
        agent.values[idx][action] += (
            reward + future - agent.values[idx][action]
        ) / agent.counts[idx][action]

        total_reward += reward
        steps += 1

    return total_reward


def nstep_temp_diff(agent, env, gamma=0.99, n=5):
    episodes = []
    done     = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        state  = env._state
        action = agent.pick_action(state)
        new_state, reward, done = env.step(action)
        episodes.append((state, action, reward))
        total_reward += reward
        steps += 1

        if len(episodes) >= n:
            state_to_update, action_to_update, _ = episodes[-n]
            idx = _idx(state_to_update)
            agent.counts[idx][action_to_update] += 1
            total = sum(gamma**i * r for i, (_, _, r) in enumerate(episodes[-n:]))
            new_idx = _idx(new_state)
            total += 0 if done else (gamma ** n) * np.max(agent.values[new_idx])
            agent.values[idx][action_to_update] += (
                total - agent.values[idx][action_to_update]
            ) / agent.counts[idx][action_to_update]

    for k in range(max(0, len(episodes) - n + 1), len(episodes)):
        state_to_update, action_to_update, _ = episodes[k]
        idx = _idx(state_to_update)
        agent.counts[idx][action_to_update] += 1
        total = sum(gamma**i * r for i, (_, _, r) in enumerate(episodes[k:]))
        agent.values[idx][action_to_update] += (
            total - agent.values[idx][action_to_update]
        ) / agent.counts[idx][action_to_update]

    return total_reward


def bview_temp_diff(agent, env, gamma=0.99, lam=0.8):
    traces = {}
    done   = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        state  = env._state
        idx    = _idx(state)
        action = agent.pick_action(state)
        new_state, reward, done = env.step(action)
        new_idx = _idx(new_state)

        agent.counts[idx][action] += 1

        prev_trace  = traces.get(idx, (action, 0))[1]
        traces[idx] = (action, prev_trace + 1)

        future = 0 if done else np.max(agent.values[new_idx])
        delta  = reward + gamma * future - agent.values[idx][action]

        for s_idx, (a, e) in traces.items():
            agent.values[s_idx][a] += (delta * e) / agent.counts[s_idx][a]
            traces[s_idx] = (a, e * gamma * lam)

        total_reward += reward
        steps += 1

    return total_reward


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
EVALUATION_STRATEGIES = {
    "Sample Average":   lambda agent, env: sample_average(agent, env),
    "First-Visit MC":   lambda agent, env: fv_monte_carlo(agent, env),
    "Every-Visit MC":   lambda agent, env: ev_monte_carlo(agent, env),
    "TD(0)":            lambda agent, env: temp_diff(agent, env),
    "N-Step TD (n=5)":  lambda agent, env: nstep_temp_diff(agent, env, n=5),
    "Backward-View TD": lambda agent, env: bview_temp_diff(agent, env),
}


def run_simulation(n_episodes=500):
    env     = MazeEnv()
    results = {}

    n_total    = len(EVALUATION_STRATEGIES)
    done_count = 0

    for eval_name, eval_fn in EVALUATION_STRATEGIES.items():
        agent = DecayingEpsilonGreedy(epsilon=1.0, decay=0.99)
        episode_rewards = []

        for ep in range(n_episodes):
            env.reset()
            reward = eval_fn(agent, env)
            episode_rewards.append(reward)
            agent.decay_epsilon()   # decay once per episode

        results[eval_name] = episode_rewards
        done_count += 1
        print(f"[{done_count}/{n_total}] {eval_name}")

    return results


# ---------------------------------------------------------------------------
# Visualization — 6 subplots, raw data, no smoothing
# ---------------------------------------------------------------------------
EVAL_COLORS = {
    "Sample Average":   "#4e9af1",
    "First-Visit MC":   "#f1954e",
    "Every-Visit MC":   "#4ef17a",
    "TD(0)":            "#f14e7a",
    "N-Step TD (n=5)":  "#c44ef1",
    "Backward-View TD": "#f1e34e",
}


def plot_results(results):
    eval_names = list(EVALUATION_STRATEGIES.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#0f1117")  # type: ignore
    fig.suptitle(
        "Evaluation Strategy Comparison  —  Decaying ε-Greedy Exploration",
        color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01
    )

    axes = axes.flatten()

    for i, eval_name in enumerate(eval_names):
        ax = axes[i]
        ax.set_facecolor("#1a1d27")
        ax.spines[:].set_color("#2e3248")
        ax.tick_params(colors="#9ca3af", labelsize=8)

        rewards = results.get(eval_name, [])
        ax.plot(np.arange(len(rewards)), rewards,
                color=EVAL_COLORS[eval_name], linewidth=0.8, alpha=0.9)

        ax.set_title(eval_name, color=EVAL_COLORS[eval_name],
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", color="#9ca3af", fontsize=8)
        ax.set_ylabel("Total Reward", color="#9ca3af", fontsize=8)
        ax.axhline(0, color="#3a3f55", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    plt.savefig("rl_results.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.show()
    print("Saved to rl_results.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N_EPISODES = 500

    print(f"Running {len(EVALUATION_STRATEGIES)} evaluation strategies "
          f"x {N_EPISODES} episodes...\n")

    results = run_simulation(n_episodes=N_EPISODES)

    print("\nPlotting results...")
    plot_results(results)
# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
# class Agent:
#     def __init__(self):
#         self.counts = np.zeros((100, 4))
#         self.values = np.zeros((100, 4))

#     def pick_action(self, state):
#         idx = state if isinstance(state, (int, np.integer)) else _idx(state)
#         return int(np.argmax(self.values[idx]))

#     def reset(self):
#         self.counts = np.zeros((100, 4))
#         self.values = np.zeros((100, 4))


# def _idx(state):
#     x, y = state
#     return y * 10 + x


# class PureExploration(Agent):
#     def pick_action(self, state):
#         return np.random.randint(4)


# class EpsilonGreedy(Agent):
#     def __init__(self, epsilon):
#         super().__init__()
#         self.epsilon = epsilon
#         self._init_epsilon = epsilon

#     def pick_action(self, state):
#         if np.random.random() >= self.epsilon:
#             return super().pick_action(state)
#         return np.random.randint(4)

#     def reset(self):
#         super().reset()
#         self.epsilon = self._init_epsilon


# class DecayingEpsilonGreedy(Agent):
#     def __init__(self, epsilon, decay):
#         super().__init__()
#         self.epsilon = epsilon
#         self.decay   = decay
#         self._init_epsilon = epsilon

#     def pick_action(self, state):
#         if np.random.random() >= self.epsilon:
#             action = super().pick_action(state)
#         else:
#             action = np.random.randint(4)
#         self.epsilon *= self.decay
#         return action

#     def reset(self):
#         super().reset()
#         self.epsilon = self._init_epsilon


# class OptimisticInitialization(Agent):
#     def __init__(self, init_value):
#         super().__init__()
#         self._init_value = init_value
#         self.values[:] = init_value

#     def reset(self):
#         super().reset()
#         self.values[:] = self._init_value


# class SoftMax(Agent):
#     def __init__(self, tao):
#         super().__init__()
#         self.tao = tao

#     def pick_action(self, state):
#         idx = _idx(state)
#         exp_v = np.exp(self.values[idx] / self.tao)
#         probs = exp_v / np.sum(exp_v)
#         return np.random.choice(4, p=probs)


# class UpperConfidenceBound(Agent):
#     def __init__(self, c):
#         super().__init__()
#         self.c = c
#         self.counts[:] = 1

#     def pick_action(self, state):
#         idx = _idx(state)
#         uncertainty = self.c * np.sqrt(
#             (2 * np.log(np.sum(self.counts[idx]))) / self.counts[idx]
#         )
#         return int(np.argmax(self.values[idx] + uncertainty))

#     def reset(self):
#         super().reset()
#         self.counts[:] = 1


# class ThompsonSampling(Agent):
#     def __init__(self, alpha, beta):
#         super().__init__()
#         self.alpha = alpha
#         self.beta  = beta
#         self.counts[:] = 1

#     def pick_action(self, state):
#         idx = _idx(state)
#         widths = self.alpha / (self.counts[idx] ** self.beta)
#         return int(np.argmax(np.random.normal(self.values[idx], widths)))

#     def reset(self):
#         super().reset()
#         self.counts[:] = 1
