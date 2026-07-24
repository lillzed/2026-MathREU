import random
from dataclasses import dataclass
from typing import Dict, Optional, Any

from . import cards
from . import encoding as enc
from .constants import (
    DEFAULT_MAX_HANDS,
    MISERE_POINTS,
    NUM_PLAYERS,
    OPEN_MISERE_POINTS,
    Phase,
    TRICKS_PER_HAND,
    WIN_POINTS,
    bid_points,
)


@dataclass
class StepResult:
    hand_completed: bool
    match_over: bool
    team_score_deltas: Optional[Dict[int, int]]
    trick_completed: bool = False
    trick_winner: Optional[int] = None
    winning_card: Optional[int] = None
    hand_summary: Optional[dict[str, Any]] = None


class FiveHundredGame:
    def __init__(self, max_hands: int =DEFAULT_MAX_HANDS) -> None:
        """
        Initialize a 500 game object.

        Input:
        - max_hands (int) : the maximum number of hands allowed in a game

        Output (None)
        """
        self.max_hands = max_hands
        self.reset()

    def reset(self, seed: int | None =None) -> None:
        """
        Reset seed, score, starting player, hand number, done, and truncated variables.
        Also deal a new hand to players, which includes shuffling the deck, dealing cards, dealing
        kitty, emptying discards, resetting all relevant round variables, stepping the hand
        number, changing the phase to bidding, and changing the current player to the starting
        player.

        Input:
        - seed (int) : the game seed

        Output (None)
        """
        self._rng = random.Random(seed)
        self.team_scores = [0, 0]
        self.start_player = 0
        self.hand_number = 0
        self.done = False
        self.truncated = False
        self._deal_new_hand()

    def _deal_new_hand(self) -> None:
        """
        Helper function for reset(). Deals a new hand.

        Input (None)

        Output (None)
        """
        deck = list(range(cards.CARD_START_INDEX, cards.JOKER_INDEX + 1))
        self._rng.shuffle(deck)
        self.hands = [set(deck[i * 10:(i + 1) * 10]) for i in range(NUM_PLAYERS)]
        self.kitty = deck[40:43]
        self.discards = []

        self.bid_history = [(0, cards.NO_SUIT)] * NUM_PLAYERS
        self.bet_winner = None
        self.highest_bet = 0
        self.highest_suit = None
        self.misere = False
        self.open_misere = False
        self.passed = [False] * NUM_PLAYERS
        self.pass_count = 0

        self.trump = None

        self.tricks_won = [0] * NUM_PLAYERS
        self.card_history: dict[int, tuple[int, int]] = {}
        self.void_suits: list[set[int]] = [set() for _ in range(NUM_PLAYERS)]
        self.current_trick = []
        self.trick_order = []
        self.trick_play_idx = 0
        self.trick_leader = None
        self.lead_suit = None
        self.sitting_out_seat = None
        self.tricks_played = 0

        self.hand_number += 1
        self.phase = Phase.BIDDING
        self.current_player = self.start_player

    def legal_actions(self, player: int  | None =None) -> list[int]:
        """
        Create a list of legal actions for the given player.

        Input:
        - player (int) : the ID of the player

        Output (list[int]):
        - the list of IDs of legal actions
        """
        if player is None:
            player = self.current_player
        if self.phase == Phase.BIDDING:
            return self._legal_bids()
        if self.phase == Phase.DISCARD:
            return list(self.hands[player])
        if self.phase == Phase.PLAY:
            return self._legal_plays(player)
        return []

    def _legal_bids(self) -> list[int]:
        """
        Helper function for legal_actions(). Create a list of legal bids.

        Input (None)

        Output (list[int]):
        - the list of IDs of legal bids
        """
        actions = [enc.bid_action()]
        hb = self.highest_bet
        hs = self.highest_suit if self.highest_suit is not None else -1

        for value in range(6, 11):
            for suit in range(5):
                if value > hb or (value == hb and suit > hs):
                    actions.append(enc.bid_action(value, suit))

        if hb == 7 and not self.misere:
            actions.append(enc.bid_action(7, misere=True))
        if (hb < 10 or (hb == 10 and hs <= cards.DIAMONDS)) and not self.open_misere:
            actions.append(enc.bid_action(10, misere=True))

        return actions

    def _legal_plays(self, player: int) -> list[int]:
        """
        Helper function for legal_actions(). Create a list of legal plays.

        Input:
        
        - player (int) : the ID of the player

        Output (list[int]):
        - the list of IDs of legal plays
        """
        hand = self.hands[player]
        if self.lead_suit is None:
            return list(hand)
        matching = [c for c in hand if cards.effective_suit(c, self.trump) == self.lead_suit]
        return matching if matching else list(hand)

    def step(self, action: int) -> StepResult:
        """
        Apply an action for the current player and advance the game by one decision,
        dispatching to the appropriate phase-specific handler (bid, discard, or card
        play) based on the current phase. The action must be one of the IDs returned
        by legal_actions() for the current player.

        Input:
        - action (int) : the ID of the action to apply

        Output (StepResult):
        - a step result object describing what happened (hand/trick/match completion,
          score deltas, trick winner, etc.)
        """
        if self.phase == Phase.BIDDING:
            return self._step_bid(action)
        if self.phase == Phase.DISCARD:
            return self._step_discard(action)
        if self.phase == Phase.PLAY:
            return self._step_play(action)
        raise RuntimeError(f"step() called while game is in terminal phase {self.phase}")

    def _step_bid(self, action: int) -> StepResult:
        """
        Helper function for step(). Step the bid.

        Input:
        - action (int) : the ID of the action

        Output (StepResult):
        - a step result object
        """
        player = self.current_player
        value, suit, misere = enc.decode_bid_action(action)

        if value == 0:
            self.passed[player] = True
            self.pass_count += 1
        elif misere:
            if value == 7:
                self.highest_bet, self.highest_suit = 7, cards.NO_SUIT
                self.misere = True
            else:
                self.highest_bet, self.highest_suit = 10, cards.DIAMONDS
                self.misere, self.open_misere = True, True
            self.bet_winner = player
        else:
            self.highest_bet, self.highest_suit = value, suit
            self.misere, self.open_misere = False, False
            self.bet_winner = player

        if value != 0 or self.bid_history[player][0] == 0:
            self.bid_history[player] = (value, suit)

        if self.pass_count == NUM_PLAYERS:
            if self.highest_bet == 0:
                self._deal_new_hand()
            else:
                self._begin_discard_phase()
            return StepResult(False, False, None)

        nxt = (player + 1) % NUM_PLAYERS
        while self.passed[nxt]:
            nxt = (nxt + 1) % NUM_PLAYERS
        self.current_player = nxt
        return StepResult(False, False, None)

    def _begin_discard_phase(self) -> None:
        """
        Helper function for step(). Sets the trump suit, adds the kitty to the players hand,
        sets the phase, the current player, and empties the discards.

        Input (None)

        Output (None)
        """
        self.trump = cards.NO_SUIT if self.open_misere else self.highest_suit
        self.hands[self.bet_winner] |= set(self.kitty)
        self.phase = Phase.DISCARD
        self.current_player = self.bet_winner
        self.discards = []

    def _step_discard(self, action: int) -> StepResult:
        """
        Helper function for step(). Step the discard.

        Input:
        - action (int) : the ID of the action

        Output (StepResult):
        - a step result object
        """
        player = self.current_player
        self.hands[player].remove(action)
        self.discards.append(action)
        if len(self.discards) == 3:
            self._begin_play_phase()
        return StepResult(False, False, None)

    def _begin_play_phase(self) -> None:
        """
        Helper function to step(). Begin play phase.

        Input (None)

        Output (None)
        """
        self.phase = Phase.PLAY
        self.tricks_played = 0
        self.trick_leader = self.bet_winner
        self.sitting_out_seat = (self.bet_winner + 2) % NUM_PLAYERS if self.misere else None
        self._begin_trick()

    def _begin_trick(self) -> None:
        """
        Helper function to step(). Begin a trick.

        Input (None)

        Output (None)
        """
        self.current_trick = []
        self.lead_suit = None
        self.trick_order = [
            (self.trick_leader + i) % NUM_PLAYERS
            for i in range(NUM_PLAYERS)
            if (self.trick_leader + i) % NUM_PLAYERS != self.sitting_out_seat
        ]
        self.trick_play_idx = 0
        self.current_player = self.trick_order[0]

    def _step_play(self, action: int) -> StepResult:
        """
        Helper function for step(). Step play.

        Input:
        - action (int) : the ID of the action

        Output (StepResult):
        - a step result object
        """
        player = self.current_player
        card = action
        self.hands[player].remove(card)
        self.current_trick.append((player, card))
        self.card_history[card] = (player, self.tricks_played)

        if len(self.current_trick) == 1:
            self.lead_suit = cards.effective_suit(card, self.trump)
            self.trick_winner = player
            self.winning_card = card
        else:
            if cards.effective_suit(card, self.trump) != self.lead_suit:
                # legal_actions() only allows this when the player held no
                # card of the lead suit, so this is a certain inference,
                # not a guess.
                self.void_suits[player].add(self.lead_suit)
            if cards.compare_cards(card, self.winning_card, self.trump, self.lead_suit) == 1:
                self.winning_card = card
                self.trick_winner = player

        self.trick_play_idx += 1
        if self.trick_play_idx < len(self.trick_order):
            self.current_player = self.trick_order[self.trick_play_idx]
            return StepResult(False, False, None)

        # trick complete
        self.tricks_won[self.trick_winner] += 1
        self.tricks_played += 1
        self.trick_leader = self.trick_winner
        trick_winner, winning_card = self.trick_winner, self.winning_card

        if self.tricks_played == TRICKS_PER_HAND:
            result = self._score_hand()
        else:
            self._begin_trick()
            result = StepResult(False, False, None)

        result.trick_completed = True
        result.trick_winner = trick_winner
        result.winning_card = winning_card
        return result


    def _score_hand(self) -> StepResult:
        """
        Helper to step(). Score a hand.
        
        Input (None)

        Output (StepResult):
        - a StepResult object
        """
        bidder = self.bet_winner
        partner = (bidder + 2) % NUM_PLAYERS
        bidder_team = bidder % 2
        other_team = 1 - bidder_team
        bidding_team_tricks = self.tricks_won[bidder] + self.tricks_won[partner]

        deltas = [0, 0]
        if self.misere:
            success = self.tricks_won[bidder] == 0
            points = OPEN_MISERE_POINTS if self.open_misere else MISERE_POINTS
            deltas[bidder_team] += points if success else -points
        else:
            success = bidding_team_tricks >= self.highest_bet
            points = bid_points(self.highest_bet, self.highest_suit)
            deltas[bidder_team] += points if success else -points
            deltas[other_team] += (TRICKS_PER_HAND - bidding_team_tricks) * 10

        hand_summary = {
            "bidder": bidder,
            "highest_bet": self.highest_bet,
            "highest_suit": self.highest_suit,
            "misere": self.misere,
            "open_misere": self.open_misere,
            "bidding_team_tricks": bidding_team_tricks,
            "success": success,
            "points": points,
        }

        self.team_scores[0] += deltas[0]
        self.team_scores[1] += deltas[1]

        match_over = any(abs(s) >= WIN_POINTS for s in self.team_scores)
        truncated = (not match_over) and self.hand_number >= self.max_hands

        if match_over or truncated:
            self.phase = Phase.GAME_OVER
            self.done = True
            self.truncated = truncated
        else:
            self.start_player = (self.start_player + 1) % NUM_PLAYERS
            self._deal_new_hand()

        return StepResult(True, match_over, {0: deltas[0], 1: deltas[1]}, hand_summary=hand_summary)
