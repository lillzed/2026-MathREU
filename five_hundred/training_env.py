import random
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices #type: ignore

from . import encoding as enc
from .constants import NUM_PLAYERS
from .hybrid_env import play_only_env #type: ignore
from .opponent_pool import HeuristicCardPolicy, OpponentPool


class FiveHundredVecEnv(VecEnv):
    """
    Adapts N independent play_only_env() AECEnvs into a single SB3 VecEnv for
    training one shared policy via self-play against a frozen opponent pool.

    Each slot plays as one randomly-chosen seat (the "learner") per episode;
    the other 3 seats are driven internally by a single opponent policy
    sampled from opponent_pool for the whole episode, so every trajectory
    follows one consistent seat's observations and rewards.
    """

    def __init__(
        self, num_envs: int, opponent_pool: OpponentPool, seed: int | None = None, **env_kwargs: Any
    ) -> None:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(enc.OBS_SIZE,), dtype=np.float32)
        action_space = spaces.Discrete(enc.ACTION_SPACE_SIZE)
        super().__init__(num_envs, observation_space, action_space)
        self.render_mode = None

        self._opponent_pool = opponent_pool
        self._rng = random.Random(seed)

        self._envs = [play_only_env(**env_kwargs) for _ in range(num_envs)]
        self._learner_agents: list[str] = [""] * num_envs
        self._opponents: list[Any] = [None] * num_envs
        self._actions = np.zeros(num_envs, dtype=np.int64)
        self._masks = np.zeros((num_envs, enc.ACTION_SPACE_SIZE), dtype=np.int8)

        for i in range(num_envs):
            self._start_new_episode(i, seed=None if seed is None else seed + i)

    def _start_new_episode(self, i: int, seed: int | None = None) -> None:
        """
        Reset env i, pick a new random learner seat and opponent snapshot for
        the episode, and fast-forward through any non-learner turns ahead of
        the learner's first turn.

        If the learner lands in a misere bidder's sitting-out seat
        (game.py's sitting_out_seat) and that hand ends the match outright,
        the match can finish before the learner ever acts; in that case
        drain the env and redeal instead of returning a dead episode.
        """
        e = self._envs[i]
        while True:
            self._learner_agents[i] = f"player_{self._rng.randrange(NUM_PLAYERS)}"
            self._opponents[i] = self._opponent_pool.sample()
            e.reset(seed=seed)
            seed = None
            _, done = self._advance_to_learner(i)
            if not done:
                return
            while e.agents:
                e.step(None)

    def _advance_to_learner(self, i: int) -> tuple[float, bool]:
        """
        Steps env i forward, playing every non-learner turn with that slot's
        sampled opponent snapshot, until either it's the learner's turn again
        or the match has ended.

        Output (tuple[float, bool]):
        - the learner's reward accumulated since their last turn
        - whether the match ended during this advance
        """
        e = self._envs[i]
        learner = self._learner_agents[i]
        opponent = self._opponents[i]

        while True:
            agent = e.agent_selection
            if e.terminations[agent] or e.truncations[agent]:
                return e._cumulative_rewards.get(learner, 0.0), True
            if agent == learner:
                return e._cumulative_rewards[learner], False
            if isinstance(opponent, HeuristicCardPolicy):
                action = opponent.act(e.unwrapped._game)
            else:
                obs = e.observe(agent)
                action, _ = opponent.predict(
                    obs["observation"], action_masks=obs["action_mask"], deterministic=False
                )
            e.step(int(action))

    def _observe_learner(self, i: int) -> np.ndarray:
        e = self._envs[i]
        obs = e.observe(self._learner_agents[i])
        self._masks[i] = obs["action_mask"]
        return obs["observation"]

    def reset(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, enc.OBS_SIZE), dtype=np.float32)
        for i in range(self.num_envs):
            obs[i] = self._observe_learner(i)
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs = np.zeros((self.num_envs, enc.OBS_SIZE), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for i, e in enumerate(self._envs):
            learner = self._learner_agents[i]
            e.step(int(self._actions[i]))
            reward, done = self._advance_to_learner(i)
            rewards[i] = reward

            if done:
                dones[i] = True
                infos[i]["terminal_observation"] = e.observe(learner)["observation"]
                while e.agents:
                    e.step(None)
                self._start_new_episode(i)

            obs[i] = self._observe_learner(i)

        return obs, rewards, dones, infos

    def close(self) -> None:
        for e in self._envs:
            e.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        indices = self._get_indices(indices)
        if attr_name == "action_masks":
            return [self._masks[i] for i in indices]
        return [getattr(self._envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for i in self._get_indices(indices):
            setattr(self._envs[i], attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: VecEnvIndices = None, **method_kwargs: Any) -> list[Any]:
        indices = self._get_indices(indices)
        if method_name == "action_masks":
            return [self._masks[i] for i in indices]
        if method_name == "add_opponent_snapshot":
            self._opponent_pool.add(*method_args, **method_kwargs)
            return [None for _ in indices]
        return [getattr(self._envs[i], method_name)(*method_args, **method_kwargs) for i in indices]

    def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
