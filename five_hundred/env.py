from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from . import encoding as enc
from . import render as trace
from .constants import DEFAULT_MAX_HANDS, NUM_PLAYERS
from .game import FiveHundredGame


class FiveHundredEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "five_hundred_v0", "is_parallelizable": False}

    def __init__(self, render_mode: str | None = None, max_hands: int = DEFAULT_MAX_HANDS) -> None:
        """
        Initialize a PettingZoo AECEnv wrapping a 500 game.

        Input:
        - render_mode (str | None) : "human" to print a play-by-play trace on reset()/step(),
          or None for no output
        - max_hands (int) : the maximum number of hands allowed before the match is truncated

        Output (None)
        """
        super().__init__()
        self.render_mode = render_mode

        self.possible_agents = [f"player_{i}" for i in range(NUM_PLAYERS)]
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}

        self._game = FiveHundredGame(max_hands=max_hands)

        self._action_spaces = {a: spaces.Discrete(enc.ACTION_SPACE_SIZE) for a in self.possible_agents}
        self._observation_spaces = {
            a: spaces.Dict({
                "observation": spaces.Box(low=0.0, high=1.0, shape=(enc.OBS_SIZE,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(enc.ACTION_SPACE_SIZE,), dtype=np.int8),
            })
            for a in self.possible_agents
        }

        self.agents = []
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.agent_selection = None

    def observation_space(self, agent: str) -> spaces.Dict:
        """
        Get the observation space for an agent.

        Input:
        - agent (str) : the agent's name, e.g. "player_0"

        Output (spaces.Dict):
        - the Dict space describing that agent's "observation" and "action_mask"
        """
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        """
        Get the action space for an agent.

        Input:
        - agent (str) : the agent's name, e.g. "player_0"

        Output (spaces.Discrete):
        - the Discrete action space shared by all agents
        """
        return self._action_spaces[agent]

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        """
        Build the current observation for an agent.

        Input:
        - agent (str) : the agent's name, e.g. "player_0"

        Output (dict[str, np.ndarray]):
        - "observation" : the encoded game state, from that agent's point of view
        - "action_mask" : legal-action mask, all zeros unless it's this agent's turn
        """
        seat = self.agent_name_mapping[agent]
        observation = enc.encode_observation(self._game, seat)
        if seat == self._game.current_player and not self._game.done:
            mask = enc.action_mask(self._game.legal_actions(seat))
        else:
            mask = np.zeros(enc.ACTION_SPACE_SIZE, dtype=np.int8)
        return {"observation": observation, "action_mask": mask}

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """
        Reset the underlying game and PettingZoo bookkeeping (agents, rewards,
        terminations/truncations, infos) for a new match.

        Input:
        - seed (int | None) : the game seed
        - options (dict | None) : unused, present for AECEnv API compatibility

        Output (None)
        """
        self._game.reset(seed=seed)
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self.possible_agents[self._game.current_player]

        self._trace_lines = [trace.describe_new_hand(self._game)]
        if self.render_mode == "human":
            self.render()

    def step(self, action: int | None) -> None:
        """
        Apply an action for the currently-selected agent, advance the underlying game,
        update rewards/terminations/truncations, and select the next agent.

        Input:
        - action (int | None) : the ID of the action to apply, or None if the current
          agent is already terminated/truncated (dead-step)

        Output (None)
        """
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0

        seat = self.agent_name_mapping[agent]
        phase_before = self._game.phase
        hand_before = self._game.hand_number
        action_line = trace.describe_action(phase_before, seat, action) if self.render_mode == "human" else None

        result = self._game.step(action)

        if self.render_mode == "human":
            lines = [action_line]
            if result.trick_completed:
                lines.append(trace.describe_trick_result(result))
            if result.hand_completed:
                lines.append(trace.describe_hand_result(result, self._game))
                if result.match_over:
                    lines.append(trace.describe_match_over(self._game))
                else:
                    lines.append(trace.describe_new_hand(self._game))
            elif self._game.hand_number != hand_before:
                lines.append(trace.describe_redeal())
                lines.append(trace.describe_new_hand(self._game))
            self._trace_lines = lines

        self.rewards = {a: 0 for a in self.agents}
        if result.team_score_deltas is not None:
            for a in self.agents:
                team = self.agent_name_mapping[a] % 2
                self.rewards[a] = result.team_score_deltas[team]

        if result.match_over or self._game.truncated:
            for a in self.agents:
                self.terminations[a] = result.match_over
                self.truncations[a] = self._game.truncated

        self.agent_selection = self.possible_agents[self._game.current_player]
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def render(self) -> None:
        """
        Print the play-by-play trace lines produced by the most recent reset()/step()
        call. No-op unless render_mode is "human".

        Input (None)

        Output (None)
        """
        if self.render_mode != "human":
            return
        for line in self._trace_lines:
            print(line)

    def close(self) -> None:
        """
        Release any resources held by the environment. No resources to release here.

        Input (None)

        Output (None)
        """
        pass


def env(**kwargs: Any) -> AECEnv:
    """
    Build a fully-wrapped 500 AECEnv, ready for use with PettingZoo-compatible agents.

    Input:
    - **kwargs (Any) : forwarded to FiveHundredEnv.__init__ (e.g. render_mode, max_hands)

    Output (AECEnv):
    - a FiveHundredEnv wrapped with PettingZoo's TerminateIllegalWrapper,
      AssertOutOfBoundsWrapper, and OrderEnforcingWrapper
    """
    e = FiveHundredEnv(**kwargs)
    e = wrappers.TerminateIllegalWrapper(e, illegal_reward=-1)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e
