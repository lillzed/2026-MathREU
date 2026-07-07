"""Pure-Python, dependency-free rules engine for this project's 500 variant.

Ports the game logic in 500/server.c and 500/cards.c into a forward
step()-able state machine (deal -> bid -> kitty/discard -> joker suit ->
play -> score -> next hand), fast enough for in-process RL training - no
subprocess, no C server involved. See five_hundred/cards.py for card/trick
rules and five_hundred/encoding.py for the action-id space this consumes.

Notable rules (see plan / server.c for citations): bidding rotates among
players who haven't passed yet and ends once all 4 have passed at some
point (the eventual winner also "passes" to decline further raises); an
all-pass hand with no bid ever made is silently redealt; the bidder's
partner sits out play entirely during misere/open misere contracts; the
joker's suit is chosen by whoever holds it after discarding, only when
trump is no-trumps.
"""

import random
from dataclasses import dataclass
from typing import Dict, Optional

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
    hand_summary: Optional[dict] = None


class FiveHundredGame:
    def __init__(self, max_hands=DEFAULT_MAX_HANDS):
        self.max_hands = max_hands
        self.reset()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        self._rng = random.Random(seed)
        self.team_scores = [0, 0]
        self.start_player = 0
        self.hand_number = 0
        self.done = False
        self.truncated = False
        self._deal_new_hand()

    def _deal_new_hand(self):
        deck = list(range(cards.NUM_CARDS))
        self._rng.shuffle(deck)
        self.hands = [set(deck[i * 10:(i + 1) * 10]) for i in range(NUM_PLAYERS)]
        self.kitty = deck[40:43]
        self.discards = []

        self.bet_winner = None
        self.highest_bet = 0
        self.highest_suit = None
        self.misere = False
        self.open_misere = False
        self.passed = [False] * NUM_PLAYERS
        self.pass_count = 0

        self.trump = None
        self.joker_suit = None

        self.tricks_won = [0] * NUM_PLAYERS
        self.seen_cards = set()
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

    # ------------------------------------------------------------------
    # legal actions
    # ------------------------------------------------------------------

    def legal_actions(self, player=None):
        if player is None:
            player = self.current_player
        if self.phase == Phase.BIDDING:
            return self._legal_bids()
        if self.phase == Phase.DISCARD:
            return list(self.hands[player])
        if self.phase == Phase.JOKER_SUIT:
            return [enc.joker_suit_action(s) for s in cards.SUITS]
        if self.phase == Phase.PLAY:
            return self._legal_plays(player)
        return []

    def _legal_bids(self):
        actions = [enc.PASS_ACTION]
        hb = self.highest_bet
        hs = self.highest_suit if self.highest_suit is not None else -1

        for value in range(6, 11):
            for suit in range(5):
                if value > hb or (value == hb and suit > hs):
                    actions.append(enc.bid_action(value, suit))

        if hb == 7 and not self.misere:
            actions.append(enc.MISERE_ACTION)
        if (hb < 10 or (hb == 10 and hs <= cards.DIAMONDS)) and not self.open_misere:
            actions.append(enc.OPENMISERE_ACTION)

        return actions

    def _legal_plays(self, player):
        hand = self.hands[player]
        if self.lead_suit is None:
            return list(hand)
        matching = [c for c in hand if cards.effective_suit(c, self.trump, self.joker_suit) == self.lead_suit]
        return matching if matching else list(hand)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action):
        if self.phase == Phase.BIDDING:
            return self._step_bid(action)
        if self.phase == Phase.DISCARD:
            return self._step_discard(action)
        if self.phase == Phase.JOKER_SUIT:
            return self._step_joker(action)
        if self.phase == Phase.PLAY:
            return self._step_play(action)
        raise RuntimeError(f"step() called while game is in terminal phase {self.phase}")

    def _step_bid(self, action):
        player = self.current_player

        if action == enc.PASS_ACTION:
            self.passed[player] = True
            self.pass_count += 1
        elif action == enc.MISERE_ACTION:
            self.highest_bet, self.highest_suit = 7, cards.NO_TRUMPS
            self.misere = True
            self.bet_winner = player
        elif action == enc.OPENMISERE_ACTION:
            self.highest_bet, self.highest_suit = 10, cards.DIAMONDS
            self.misere, self.open_misere = True, True
            self.bet_winner = player
        else:
            value, suit = enc.decode_bid_action(action)
            self.highest_bet, self.highest_suit = value, suit
            self.misere, self.open_misere = False, False
            self.bet_winner = player

        if self.pass_count == NUM_PLAYERS:
            if self.highest_bet == 0:
                # everyone passed without a single bid - reshuffle and redeal,
                # invisible to the episode (server.c: "Everyone passed" -> game_loop again)
                self._deal_new_hand()
            else:
                self._begin_discard_phase()
            return StepResult(False, False, None)

        nxt = (player + 1) % NUM_PLAYERS
        while self.passed[nxt]:
            nxt = (nxt + 1) % NUM_PLAYERS
        self.current_player = nxt
        return StepResult(False, False, None)

    def _begin_discard_phase(self):
        # open misere is ranked using DIAMONDS as a placeholder (see _legal_bids /
        # server.c bet_round) but always plays as no-trumps
        self.trump = cards.NO_TRUMPS if self.open_misere else self.highest_suit
        self.hands[self.bet_winner] |= set(self.kitty)
        self.phase = Phase.DISCARD
        self.current_player = self.bet_winner
        self.discards = []

    def _step_discard(self, action):
        player = self.current_player
        self.hands[player].remove(action)
        self.discards.append(action)
        if len(self.discards) == 3:
            self._begin_joker_or_play_phase()
        return StepResult(False, False, None)

    def _begin_joker_or_play_phase(self):
        if self.trump == cards.NO_TRUMPS:
            holder = next((s for s in range(NUM_PLAYERS) if cards.JOKER in self.hands[s]), None)
            if holder is not None:
                self.phase = Phase.JOKER_SUIT
                self.current_player = holder
                return
        self._begin_play_phase()

    def _step_joker(self, action):
        self.joker_suit = enc.decode_joker_suit_action(action)
        self._begin_play_phase()
        return StepResult(False, False, None)

    def _begin_play_phase(self):
        self.phase = Phase.PLAY
        self.tricks_played = 0
        self.trick_leader = self.bet_winner
        self.sitting_out_seat = (self.bet_winner + 2) % NUM_PLAYERS if self.misere else None
        self._begin_trick()

    def _begin_trick(self):
        self.current_trick = []
        self.lead_suit = None
        self.trick_order = [
            (self.trick_leader + i) % NUM_PLAYERS
            for i in range(NUM_PLAYERS)
            if (self.trick_leader + i) % NUM_PLAYERS != self.sitting_out_seat
        ]
        self.trick_play_idx = 0
        self.current_player = self.trick_order[0]

    def _step_play(self, action):
        player = self.current_player
        card = action
        self.hands[player].remove(card)
        self.current_trick.append((player, card))
        self.seen_cards.add(card)

        if len(self.current_trick) == 1:
            self.lead_suit = cards.effective_suit(card, self.trump, self.joker_suit)
            self.trick_winner = player
            self.winning_card = card
        elif cards.compare_cards(card, self.winning_card, self.trump, self.joker_suit) == 1:
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

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _score_hand(self):
        bidder = self.bet_winner
        partner = (bidder + 2) % NUM_PLAYERS
        bidder_team = bidder % 2
        other_team = 1 - bidder_team
        bidding_team_tricks = self.tricks_won[bidder] + self.tricks_won[partner]

        deltas = [0, 0]
        if self.misere:
            # partner never plays during misere, so their trick count is always 0 -
            # this check is equivalent to server.c's get_winning_tricks(game) == 0
            success = self.tricks_won[bidder] == 0
            points = OPEN_MISERE_POINTS if self.open_misere else MISERE_POINTS
            deltas[bidder_team] += points if success else -points
        else:
            success = bidding_team_tricks >= self.highest_bet
            points = bid_points(self.highest_bet, self.highest_suit)
            deltas[bidder_team] += points if success else -points
            deltas[other_team] += (TRICKS_PER_HAND - bidding_team_tricks) * 10

        # captured before _deal_new_hand() (below) resets bet_winner/highest_bet/etc
        # for the next hand, so callers (e.g. render.py) can still describe this
        # just-finished hand's contract after the engine has already moved on
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
