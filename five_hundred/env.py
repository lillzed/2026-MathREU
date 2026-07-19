from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from . import cards
from . import encoding as enc
from . import render as trace
from .constants import DEFAULT_MAX_HANDS, NUM_PLAYERS, Phase, TRICKS_PER_HAND
from .game import FiveHundredGame

# penalty for winning the trick-so-far with a higher card than needed;
# trick rewards alone don't distinguish which card won
_WASTE_PENALTY = 0.05


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
        self._outcome_decided = False

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
        self._outcome_decided = False

        self._trace_lines = [trace.describe_new_hand(self._game)]
        if self.render_mode == "human":
            self.render()

    def _contract_ranks(self, misere: bool, open_misere: bool, highest_bet: int) -> tuple[float, float]:
        """
        How many tricks each side needs won for the hand to go in their own
        favor: (attacking team's rank, defending team's rank). Lower rank
        means a bigger +-1/rank swing, i.e. higher stakes.

        Input:
        - misere (bool) : whether the contract is misere/open misere
        - open_misere (bool) : whether the misere is specifically open misere
          (ignored unless misere is True)
        - highest_bet (int) : the winning bid's value (ignored for misere)

        Output (tuple[float, float]):
        - attacking team's rank, defending team's rank
        """
        if misere:
            # open misere pays OPEN_MISERE_POINTS (500) vs. regular misere's
            # MISERE_POINTS (250) -- halve both ranks so the reward swing is
            # twice as large, matching the real stakes, instead of scoring an
            # open misere identically to a regular one.
            return (5, 0.5) if open_misere else (10, 1)
        return highest_bet, 11 - highest_bet

    def _trick_reward(
        self, bet_winner: int, misere: bool, open_misere: bool, highest_bet: int, trick_winner: int
    ) -> dict[str, float]:
        """
        Per-trick reward for the trick that was just completed, keyed by
        agent name. Whichever team's success condition just advanced gets
        +1/their_rank, the other team gets -1/their_rank. For misere, the
        team whose condition advances is the *opposite* of whoever physically
        won the trick: the bidder winning a trick is bad for them (it's the
        defenders' condition that advanced), and vice versa.

        Input:
        - bet_winner (int) : the seat ID of the hand's bidder
        - misere (bool) : whether the contract is misere/open misere
        - open_misere (bool) : whether the misere is specifically open misere
        - highest_bet (int) : the winning bid's value
        - trick_winner (int) : the seat ID of the player who won the trick

        Output (dict[str, float]):
        - reward for this trick, keyed by agent name
        """
        attacker_rank, defender_rank = self._contract_ranks(misere, open_misere, highest_bet)
        attacker_team = bet_winner % 2
        physical_winner_team = trick_winner % 2

        benefiting_team = (1 - physical_winner_team) if misere else physical_winner_team

        rank = attacker_rank if benefiting_team == attacker_team else defender_rank
        reward = 1.0 / rank

        return {
            a: reward if self.agent_name_mapping[a] % 2 == benefiting_team else -reward
            for a in self.agents
        }

    def _card_efficiency_penalty(
        self,
        played: int,
        hand_before: set[int],
        trick_before: list[tuple[int, int]],
        trump: int,
        lead_suit: int | None,
    ) -> float:
        """
        Penalize beating the trick-so-far by more than necessary: if a
        strictly cheaper card from the same (pre-play) hand would also have
        won against the best card on the table, this play burned a better
        card for no reason.

        Input:
        - played (int) : the card the current player just played
        - hand_before (set[int]) : their hand before playing (includes `played`)
        - trick_before (list[tuple[int, int]]) : (player, card) pairs already
          played this trick before this action; empty if this play led the trick
        - trump (int) : this hand's trump suit
        - lead_suit (int | None) : the trick's lead suit, or None if this play led it

        Output (float):
        - 0.0 if there was nothing to overtake or no cheaper card would have
          done the job, else -_WASTE_PENALTY
        """
        if not trick_before:
            return 0.0

        best_card = trick_before[0][1]
        for _, card in trick_before[1:]:
            if cards.compare_cards(card, best_card, trump, lead_suit) == 1:
                best_card = card

        if cards.compare_cards(played, best_card, trump, lead_suit) != 1:
            return 0.0

        played_rank = cards.effective_card(played, trump)[0]
        cheaper_alternatives = [
            c for c in hand_before
            if c != played
            and cards.effective_card(c, trump)[0] < played_rank
            and cards.compare_cards(c, best_card, trump, lead_suit) == 1
        ]
        return -_WASTE_PENALTY if cheaper_alternatives else 0.0

    def _outcome_certain(
        self,
        bet_winner: int,
        misere: bool,
        highest_bet: int,
        tricks_won: list[int],
        tricks_played: int,
    ) -> tuple[bool, int | None]:
        """
        Whether the hand's outcome is already mathematically decided given
        tricks played so far, and if so, which team it favors.

        Input:
        - bet_winner (int) : the seat ID of the hand's bidder
        - misere (bool) : whether the contract is misere/open misere
        - highest_bet (int) : the winning bid's value
        - tricks_won (list[int]) : tricks won per seat, after the trick that just completed
        - tricks_played (int) : tricks completed so far this hand, after the trick that just completed

        Output (tuple[bool, int | None]):
        - whether the outcome is decided
        - the team it favors, or None if not yet decided
        """
        attacker_team = bet_winner % 2

        if misere:
            if tricks_won[bet_winner] >= 1:
                return True, 1 - attacker_team
            if tricks_played == TRICKS_PER_HAND:
                return True, attacker_team
            return False, None

        partner = (bet_winner + 2) % NUM_PLAYERS
        attacker_tricks = tricks_won[bet_winner] + tricks_won[partner]
        if attacker_tricks >= highest_bet:
            return True, attacker_team
        remaining = TRICKS_PER_HAND - tricks_played
        if attacker_tricks + remaining < highest_bet:
            return True, 1 - attacker_team
        return False, None

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

        # a trick-completing action can trigger _deal_new_hand() internally
        # (game.py's _score_hand()), which resets bet_winner/tricks_won/etc.
        # before step() returns -- so the contract state has to be captured
        # here, before stepping, not read back off self._game afterward.
        bet_winner_before = self._game.bet_winner
        misere_before = self._game.misere
        open_misere_before = self._game.open_misere
        highest_bet_before = self._game.highest_bet
        tricks_won_before = list(self._game.tricks_won)
        tricks_played_before = self._game.tricks_played

        play_hand_before = None
        if phase_before == Phase.PLAY:
            play_hand_before = set(self._game.hands[seat])
            trick_before = list(self._game.current_trick)
            trump_before = self._game.trump
            lead_suit_before = self._game.lead_suit

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

        self.rewards = {a: 0.0 for a in self.agents}
        # card values invert under misere (low cards are the ones worth
        # keeping), so the efficiency penalty only applies to standard contracts
        if play_hand_before is not None and not misere_before:
            self.rewards[agent] += self._card_efficiency_penalty(
                action, play_hand_before, trick_before, trump_before, lead_suit_before
            )

        if result.trick_completed:
            trick_reward = self._trick_reward(
                bet_winner_before, misere_before, open_misere_before, highest_bet_before, result.trick_winner
            )
            for a, r in trick_reward.items():
                self.rewards[a] += r

            tricks_won_after = list(tricks_won_before)
            tricks_won_after[result.trick_winner] += 1
            decided, favored_team = self._outcome_certain(
                bet_winner_before, misere_before, highest_bet_before,
                tricks_won_after, tricks_played_before + 1,
            )
            if decided and not self._outcome_decided:
                self._outcome_decided = True
                for a in self.agents:
                    team = self.agent_name_mapping[a] % 2
                    self.rewards[a] += 1.0 if team == favored_team else -1.0

        if result.hand_completed:
            self._outcome_decided = False

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
