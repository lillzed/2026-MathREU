import random
from multiprocessing import Pool
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices #type: ignore

from . import encoding as enc
from .constants import NUM_PLAYERS
from .hybrid_env import play_only_env #type: ignore
from .mc_bot import MonteCarloCardPolicy, choose_card_worker
from .opponent_pool import OpponentPool


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
        self,
        num_envs: int,
        opponent_pool: OpponentPool,
        seed: int | None = None,
        opponent_workers: int = 1,
        **env_kwargs: Any,
    ) -> None:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(enc.OBS_SIZE,), dtype=np.float32)
        action_space = spaces.Discrete(enc.ACTION_SPACE_SIZE)
        super().__init__(num_envs, observation_space, action_space)
        self.render_mode = None

        self._opponent_pool = opponent_pool
        self._rng = random.Random(seed)
        self._task_seed_rng = random.Random(seed)
        self._pool = Pool(processes=opponent_workers) if opponent_workers > 1 else None

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
            if hasattr(opponent, "act"):
                action = opponent.act(e.unwrapped._game)
            else:
                obs = e.observe(agent)
                action, _ = opponent.predict(
                    obs["observation"], action_masks=obs["action_mask"], deterministic=False
                )
            e.step(int(action))

    def _advance_all(self, indices: list[int]) -> tuple[list[float], list[bool]]:
        """
        Batched version of _advance_to_learner covering every env in
        `indices` at once: each round, every env still short of its
        learner's turn contributes at most one pending decision, and any of
        those backed by a MonteCarloCardPolicy anchor are dispatched to
        self._pool together instead of one at a time. Determinization
        counts strong enough to beat heuristic_play.py cost double-digit
        milliseconds per decision (see mc_bot.py), and every non-learner
        seat needs one every step, so computing them one env at a time would
        make training throughput scale with a single core no matter how
        many CPUs the machine has.

        Envs are independent, so resolving them in interleaved rounds
        instead of one full env at a time changes nothing about the result
        -- only how the work is batched.

        Output (tuple[list[float], list[bool]]):
        - reward accumulated since each env's learner last acted
        - whether each env's match ended during this advance
        """
        pending = list(indices)
        rewards = {i: 0.0 for i in indices}
        done = {i: False for i in indices}

        while pending:
            pool_tasks: list[tuple[Any, int, int]] = []
            pool_targets: list[int] = []
            still_pending: list[int] = []

            for i in pending:
                e = self._envs[i]
                learner = self._learner_agents[i]
                agent = e.agent_selection

                if e.terminations[agent] or e.truncations[agent]:
                    rewards[i] = e._cumulative_rewards.get(learner, 0.0)
                    done[i] = True
                    continue
                if agent == learner:
                    rewards[i] = e._cumulative_rewards[learner]
                    continue

                opponent = self._opponents[i]
                if self._pool is not None and isinstance(opponent, MonteCarloCardPolicy):
                    seed = self._task_seed_rng.randrange(2**31)
                    pool_tasks.append((e.unwrapped._game, opponent.num_determinizations, seed))
                    pool_targets.append(i)
                elif hasattr(opponent, "act"):
                    e.step(int(opponent.act(e.unwrapped._game)))
                else:
                    obs = e.observe(agent)
                    action, _ = opponent.predict(
                        obs["observation"], action_masks=obs["action_mask"], deterministic=False
                    )
                    e.step(int(action))
                still_pending.append(i)

            if pool_tasks:
                for i, action in zip(pool_targets, self._pool.map(choose_card_worker, pool_tasks)):
                    self._envs[i].step(int(action))

            pending = still_pending

        return [rewards[i] for i in indices], [done[i] for i in indices]

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
            e.step(int(self._actions[i]))

        step_rewards, step_dones = self._advance_all(list(range(self.num_envs)))

        for i, e in enumerate(self._envs):
            rewards[i] = step_rewards[i]
            if step_dones[i]:
                dones[i] = True
                infos[i]["terminal_observation"] = e.observe(self._learner_agents[i])["observation"]
                while e.agents:
                    e.step(None)
                self._start_new_episode(i)
            obs[i] = self._observe_learner(i)

        return obs, rewards, dones, infos

    def close(self) -> None:
        for e in self._envs:
            e.close()
        if self._pool is not None:
            self._pool.close()
            self._pool.join()

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
