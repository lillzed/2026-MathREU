from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices #type: ignore

from . import encoding as enc
from .hybrid_env import play_only_env #type: ignore


class FiveHundredVecEnv(VecEnv):
    """
    Adapts N independent play_only_env() AECEnvs into a single SB3 VecEnv for
    training one shared policy via self-play. Each slot's "agent identity"
    rotates through whichever of the 4 seats is currently up
    """

    def __init__(self, num_envs: int, seed: int | None = None, **env_kwargs: Any) -> None:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(enc.OBS_SIZE,), dtype=np.float32)
        action_space = spaces.Discrete(enc.ACTION_SPACE_SIZE)
        super().__init__(num_envs, observation_space, action_space)
        self.render_mode = None

        self._envs = [play_only_env(**env_kwargs) for _ in range(num_envs)]
        self._actions = np.zeros(num_envs, dtype=np.int64)
        self._masks = np.zeros((num_envs, enc.ACTION_SPACE_SIZE), dtype=np.int8)

        for i, e in enumerate(self._envs):
            e.reset(seed=None if seed is None else seed + i)

    def _observe_current(self, i: int) -> np.ndarray:
        e = self._envs[i]
        obs = e.observe(e.agent_selection)
        self._masks[i] = obs["action_mask"]
        return obs["observation"]

    def reset(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, enc.OBS_SIZE), dtype=np.float32)
        for i in range(self.num_envs):
            obs[i] = self._observe_current(i)
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs = np.zeros((self.num_envs, enc.OBS_SIZE), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for i, e in enumerate(self._envs):
            e.step(int(self._actions[i]))

            agent = e.agent_selection
            terminal = e.terminations[agent] or e.truncations[agent]
            _, reward, *_ = e.last()
            rewards[i] = reward

            if terminal:
                dones[i] = True
                infos[i]["terminal_observation"] = e.observe(agent)["observation"]
                while e.agents:
                    e.step(None)
                e.reset()

            obs[i] = self._observe_current(i)

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
        return [getattr(self._envs[i], method_name)(*method_args, **method_kwargs) for i in indices]

    def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
