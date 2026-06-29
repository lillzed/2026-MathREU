import math
import random
import numpy as np
import plotly.graph_objects as go

#____________________Environment____________________#
class MultiArmedBandit:
    def __init__(self, min_arm: int, max_arm: int, best_arm: int, variance: float, reward_noise: float) -> None:
        assert min_arm <= best_arm and best_arm <= max_arm
        
        self.min_arm = min_arm
        self.max_arm = max_arm
        self.best_arm = best_arm
        self.num_arms = max_arm - min_arm + 1

        self.variance = variance
        self.reward_noise = reward_noise
    
    def pull(self, arm: int) -> float:
        assert self.min_arm <= arm and arm <= self.max_arm

        reward = math.e ** (-((arm - self.best_arm) ** 2)/(2 * self.variance))
        return reward + random.gauss(0, self.reward_noise)

#_____________________Models_______________________#
class Agent:
    def __init__(self, env: MultiArmedBandit) -> None:
        self.env = env

        self.counts = np.zeros(self.env.num_arms)
        self.values = np.zeros(self.env.num_arms)
    
    def select_arm(self) -> int:
        return int(np.argmax(self.values) + self.env.min_arm)
    
    def update(self, arm: int, reward: float) -> None:
        i = arm - self.env.min_arm

        self.counts[i] += 1
        self.values[i] += (reward - self.values[i]) / self.counts[i]

class PureExploitation(Agent):
    pass
#____________________________
class PureExploration(Agent):
    def select_arm(self) -> int:
        return random.randint(self.env.min_arm, self.env.max_arm)
#_____________________________
class EpsilonGreedy(Agent):
    def __init__(self, env: MultiArmedBandit, epsilon: float) -> None:
        super().__init__(env)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(self.env.min_arm, self.env.max_arm)
        else:
            return super().select_arm()
#_____________________________
class DecayingEpsilonGreedy(Agent):
    def __init__(self, env: MultiArmedBandit, epsilon: float, decay_rate: float) -> None:
        super().__init__(env)
        self.epsilon = epsilon
        self.decay_rate = decay_rate
    
    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(self.env.min_arm, self.env.max_arm)
        else:
            return super().select_arm()
    
    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        self.epsilon *= (.5 ** (1 / self.decay_rate))
#_________________________
class OptimisticInitialization(Agent):
    def __init__(self, env: MultiArmedBandit, start_value: float, start_count: int) -> None:
        super().__init__(env)
        self.counts[:] = start_count
        self.values[:] = start_value
#_________________________
class SoftMax(Agent):
    def __init__(self, env: MultiArmedBandit, tao: float) -> None:
        super().__init__(env)
        self.tao = tao
        self.exp_q = np.exp(self.values / self.tao)
        self.probabilities = self.exp_q / np.sum(self.exp_q)
    
    def select_arm(self) -> int:
        return np.random.choice(self.env.num_arms, p=self.probabilities) + self.env.min_arm
    
    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        i = arm - self.env.min_arm
        self.exp_q[i] = np.exp(self.values[i] / self.tao)
        self.probabilities = self.exp_q / np.sum(self.exp_q)
#___________________________
class UpperConfidenceBound(Agent):
    def __init__(self, env: MultiArmedBandit, start_count: int, c: float) -> None:
        super().__init__(env)
        self.counts[:] = start_count
        self.c = c
    
    def select_arm(self) -> int:
        return int(np.argmax(self.values + self.c * np.sqrt(np.log(np.sum(self.counts)) / self.counts))) + self.env.min_arm
#____________________________
class ThompsonSampling(Agent):
    def __init__(self, env: MultiArmedBandit, alpha: float, beta: float) -> None:
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta

        self.counts[:] = 1
        self.widths = self.alpha / (self.counts ** self.beta)
    
    def select_arm(self) -> int:
        return int(np.argmax(np.random.normal(self.values, self.widths)) + self.env.min_arm)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        i = arm - self.env.min_arm
        self.widths[i] = self.alpha / (self.counts[i] ** self.beta)

#__________________Simulation_____________________#
def run_simulation(agents: dict, env: MultiArmedBandit, steps: int = 1000) -> tuple[dict, dict]:
    results = {name: [] for name in agents}
    cumulative = {name: 0.0 for name in agents}
    cumulative_regret  = {name: 0.0 for name in agents}
    regrets = {name: [] for name in agents}

    for step in range(1, steps + 1):
        for name, agent in agents.items():
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)
            cumulative[name] += reward
            results[name].append(cumulative[name] / step)
            cumulative_regret[name] += (1 - reward)
            regrets[name].append(cumulative_regret[name] / step)

    return results, regrets

#__________________Run_____________________#
env = MultiArmedBandit(min_arm=-100, max_arm=100, best_arm=0, variance=50, reward_noise=0.7)

agents = {
    "Pure Exploitation":         PureExploitation(env),
    "Pure Exploration":          PureExploration(env),
    "Epsilon Greedy":            EpsilonGreedy(env, epsilon=0.1),
    "Decaying Epsilon":          DecayingEpsilonGreedy(env, epsilon=0.5, decay_rate=500),
    "UCB":                       UpperConfidenceBound(env, start_count=1, c=1.0),
    "Thompson Sampling":         ThompsonSampling(env, alpha=1.0, beta=0.5),
    "Softmax":                   SoftMax(env, tao=.1),
    "Optimistic Initialization": OptimisticInitialization(env, start_value=1.0, start_count=1),
}

results = run_simulation(agents, env, steps=100000)

fig = go.Figure()
for name, rewards in results[0].items():
    fig.add_trace(go.Scatter(y=rewards, mode='lines', name=name))
for name, regrets in results[1].items():
    fig.add_trace(go.Scatter(y=regrets, mode='lines', name= f"{name} Regrets"))

fig.update_layout(
    title="Multi-Armed Bandit Comparison",
    xaxis_title="Step",
    yaxis_title="Average Reward/Regret"
)
fig.show()