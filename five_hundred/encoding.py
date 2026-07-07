"""Unified action-id space and observation encoding shared by game.py and env.py.

Action space (Discrete, size ACTION_SPACE_SIZE):
    0..42    card id - used as the action during DISCARD and PLAY phases
    43       PASS       (bid phase only)
    44       MISERE     (bid phase only)
    45       OPENMISERE (bid phase only)
    46..70   normal bid: value in 6..10, suit in 0..4 (S,C,D,H,N)
    71..74   joker suit choice: suit in 0..3 (S,C,D,H)      (JOKER_SUIT phase only)
"""

import numpy as np

from . import cards
from .constants import NUM_PLAYERS, TRICKS_PER_HAND, Phase

PASS_ACTION = cards.NUM_CARDS          # 43
MISERE_ACTION = PASS_ACTION + 1        # 44
OPENMISERE_ACTION = PASS_ACTION + 2    # 45
BID_ACTION_BASE = PASS_ACTION + 3      # 46, 5 values x 5 suits = 25 actions
JOKER_SUIT_ACTION_BASE = BID_ACTION_BASE + 25  # 71, 4 actions
ACTION_SPACE_SIZE = JOKER_SUIT_ACTION_BASE + 4  # 75


def bid_action(value, suit):
    return BID_ACTION_BASE + (value - 6) * 5 + suit


def decode_bid_action(action):
    offset = action - BID_ACTION_BASE
    return 6 + offset // 5, offset % 5


def joker_suit_action(suit):
    return JOKER_SUIT_ACTION_BASE + suit


def decode_joker_suit_action(action):
    return action - JOKER_SUIT_ACTION_BASE


def action_mask(legal_actions):
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    mask[list(legal_actions)] = 1
    return mask


# ---------------------------------------------------------------------------
# Observation encoding - seat-relative: for a viewer at seat `viewer`, relative
# index r (0..3) always refers to actual seat (viewer + r) % NUM_PLAYERS, so
# 0 = self, 2 = partner, 1/3 = opponents. Keeps the 4 seats' observations
# symmetric for self-play.
# ---------------------------------------------------------------------------

_PHASES = (Phase.BIDDING, Phase.DISCARD, Phase.JOKER_SUIT, Phase.PLAY, Phase.GAME_OVER)
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASES)}

# block sizes, in the order encode_observation builds them - see below
_BLOCK_SIZES = [
    cards.NUM_CARDS,       # 1. own hand
    len(_PHASES),          # 2. phase one-hot
    6,                     # 3. trump one-hot (S,C,D,H,N,none)
    5,                     # 4. joker suit one-hot (S,C,D,H,none)
    1,                     # 5. highest bid value, normalized
    6,                     # 6. highest bid suit one-hot (S,C,D,H,N,none)
    2,                     # 7. misere / open-misere flags
    NUM_PLAYERS,           # 8. passed[], relative
    5,                     # 9. current highest bidder, relative one-hot (+none)
    NUM_PLAYERS,           # 10. dealer/start seat, relative one-hot
    *([cards.NUM_CARDS + 1] * NUM_PLAYERS),  # 11. cards on table this trick, per relative seat (+none)
    cards.NUM_CARDS,       # 12. cards seen in earlier tricks this hand
    cards.NUM_CARDS,       # 13. revealed hand (open misere bidder, once play starts)
    2,                     # 14. own/opponent team tricks won this hand, normalized
]
OBS_SIZE = sum(_BLOCK_SIZES)


def _rel_seat(actual_seat, viewer):
    return (actual_seat - viewer) % NUM_PLAYERS


def _onehot(index, size):
    v = np.zeros(size, dtype=np.float32)
    if index is not None:
        v[index] = 1.0
    return v


def encode_observation(game, viewer):
    blocks = []

    # 1. own hand
    own_hand = np.zeros(cards.NUM_CARDS, dtype=np.float32)
    for c in game.hands[viewer]:
        own_hand[c] = 1.0
    blocks.append(own_hand)

    # 2. phase one-hot
    blocks.append(_onehot(_PHASE_INDEX.get(game.phase), len(_PHASES)))

    # 3. trump one-hot (index 5 = not decided yet)
    blocks.append(_onehot(game.trump if game.trump is not None else 5, 6))

    # 4. joker suit one-hot (index 4 = not chosen / not applicable)
    blocks.append(_onehot(game.joker_suit if game.joker_suit is not None else 4, 5))

    # 5. highest bid value, normalized
    blocks.append(np.array([game.highest_bet / 10.0], dtype=np.float32))

    # 6. highest bid suit one-hot (index 5 = none yet)
    blocks.append(_onehot(game.highest_suit if game.highest_suit is not None else 5, 6))

    # 7. misere / open misere flags
    blocks.append(np.array([float(game.misere), float(game.open_misere)], dtype=np.float32))

    # 8. passed[], relative
    passed_rel = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for r in range(NUM_PLAYERS):
        passed_rel[r] = float(game.passed[(viewer + r) % NUM_PLAYERS])
    blocks.append(passed_rel)

    # 9. current highest bidder, relative one-hot (index 4 = no bid yet)
    bidder_idx = 4 if game.bet_winner is None else _rel_seat(game.bet_winner, viewer)
    blocks.append(_onehot(bidder_idx, 5))

    # 10. dealer/start seat, relative one-hot
    blocks.append(_onehot(_rel_seat(game.start_player, viewer), NUM_PLAYERS))

    # 11. cards on the table this trick, per relative seat (index NUM_CARDS = none played)
    table = dict(game.current_trick)
    for r in range(NUM_PLAYERS):
        actual = (viewer + r) % NUM_PLAYERS
        card = table.get(actual)
        slot = np.zeros(cards.NUM_CARDS + 1, dtype=np.float32)
        slot[card if card is not None else cards.NUM_CARDS] = 1.0
        blocks.append(slot)

    # 12. cards seen in earlier tricks this hand
    seen = np.zeros(cards.NUM_CARDS, dtype=np.float32)
    for c in game.seen_cards:
        seen[c] = 1.0
    blocks.append(seen)

    # 13. revealed hand: the open-misere bidder's hand is public once play starts
    revealed = np.zeros(cards.NUM_CARDS, dtype=np.float32)
    if game.open_misere and game.phase == Phase.PLAY and game.bet_winner is not None:
        for c in game.hands[game.bet_winner]:
            revealed[c] = 1.0
    blocks.append(revealed)

    # 14. own/opponent team tricks won this hand, normalized
    own_team = viewer % 2
    own_tricks = sum(game.tricks_won[s] for s in range(NUM_PLAYERS) if s % 2 == own_team)
    opp_tricks = sum(game.tricks_won[s] for s in range(NUM_PLAYERS) if s % 2 != own_team)
    blocks.append(np.array([own_tricks / TRICKS_PER_HAND, opp_tricks / TRICKS_PER_HAND], dtype=np.float32))

    return np.concatenate(blocks)
