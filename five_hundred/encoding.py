"""
Actions:
    0..42   4 Hearts - Joker (Cards)
    43      Pass
    44..68  6 Spade - 10 No Trump (Bids)
    69      Misere
    70      Open Misere
    71..74  Spades - Hearts (Pick Joker Suit)
"""

import numpy as np
import numpy.typing as npt
from typing import Iterable, TYPE_CHECKING
from . import cards
from . constants import Phase, NUM_PLAYERS, TRICKS_PER_HAND

if TYPE_CHECKING:
    # game.py imports this module at load time, so importing FiveHundredGame here
    # for anything but a type hint would be a circular import.
    from .game import FiveHundredGame

ACTION_SPACE_SIZE = 75
BID_BASE = 44
PASS = 43
MISERE = 69
OPEN_MISERE = 70
JOKER_SUIT_BASE = 71
SPADES, CLUBS, DIAMONDS, HEARTS, NO_SUIT = 0, 1, 2, 3, 4
NUM_CARDS = cards.JOKER_INDEX - cards.CARD_START_INDEX + 1

def bid_action(value: int = 0, suit: int = NO_SUIT, misere: bool = False) -> int:
    """
    Returns the ID of a bid action from the value and suit. An optional parameter for
    misere is provided. Suit should be set to NO_SUIT and value should be set to 7 to designate
    misere while a value of 10 should be used for open misere. A value of 0 will default to a
    pass action.
    """
    if value == 0:
        return PASS
    if misere:
        return MISERE if value == 7 else OPEN_MISERE
    return BID_BASE + ((value - 6) * 5) + suit


def decode_bid_action(id: int) -> tuple[int, int, bool]:
    """
    Returns a tuple of [value, suit, misere], given the ID of a bid.
    Passing returns (0, NO_SUIT, False). Misere returns (7, NO_SUIT, True).
    Open Misere returns (10, NO_SUIT, True).
    """
    if id == PASS:
        return (0, NO_SUIT, False)
    elif id == MISERE:
        return (7, NO_SUIT, True)
    elif id == OPEN_MISERE:
        return (10, NO_SUIT, True)
    return ((id - BID_BASE + 30) // 5, (id - BID_BASE) % 5, False)


def joker_suit_action(suit: int) -> int:
    """Returns the ID of choosing the given suit for the joker."""
    return JOKER_SUIT_BASE + suit


def decode_joker_suit_action(id: int) -> int:
    """Returns the suit of the joker given the ID for assigning the joker to that suit."""
    return (JOKER_SUIT_BASE - id) % 5


def action_mask(legal_actions: Iterable[int]) -> npt.NDArray[np.int8]:
    """Returns a mask over all actions from a list of legal actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    mask[list(legal_actions)] = 1
    return mask

def rel_seat(viewer: int, player: int) -> int:
    """Returns the relative seat ID of a player, given the IDs of the viewer's and player's seat."""
    return (viewer - player) % 4


def one_hot(index: int | None, size: int) -> npt.NDArray[np.float32]:
    result = np.zeros(size, dtype=np.float32)
    if index is not None:
        result[index] = 1.0
    return result


PHASES = (Phase.BIDDING, Phase.DISCARD, Phase.PLAY, Phase.GAME_OVER)
PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}

BLOCK_SIZES = [
    NUM_CARDS,              # 1. own hand
    len(PHASES),            # 2. phase
    6,                      # 3. trump (S,C,D,H,N,none)
    NUM_PLAYERS,            # 4. highest bid rank per player, relative, normalized
    NUM_PLAYERS * 6,        # 5. highest bid suit per player, relative (S,C,D,H,N,hasn't bid)
    2,                      # 6. misere / open-misere flags
    NUM_PLAYERS,            # 7. passed[], relative
    5,                      # 8. current highest bidder, relative (+none)
    NUM_PLAYERS,            # 9. dealer/start seat, relative
    (NUM_CARDS + 1) * NUM_PLAYERS,  # 10. cards on table this trick, per relative seat (+none)
    NUM_CARDS * 6,          # 11. card history (card, who played (+none) and what trick)
    NUM_CARDS,              # 12. revealed hand (open misere bidder, once play starts)
    2,                      # 13. own/opponent team tricks won this hand, normalized
]
OBS_SIZE = sum(BLOCK_SIZES)


def encode_observation(game: "FiveHundredGame", viewer: int) -> np.ndarray:
    """
    Encodes a game state into a flat observation from one player's point of view. Concatenates,
    in order: own hand, phase, trump, highest bid rank/suit per player (relative), misere/open
    misere flags, passed[] (relative), current highest bidder (relative), dealer seat (relative),
    cards on the table this trick (relative), full card-play history (who played each card and
    in which trick, relative), the revealed hand of an open misere bidder, and tricks won this
    hand (own/opponent).
    """
    blocks: list[np.ndarray] = []

    # 1. own hand
    own_hand = np.zeros(NUM_CARDS, dtype=np.float32)
    for c in game.hands[viewer]:
        own_hand[c] = 1.0
    blocks.append(own_hand)

    # 2. phase
    blocks.append(one_hot(PHASE_INDEX.get(game.phase), len(PHASES)))

    # 3. trump (index 5 = not chosen yet)
    blocks.append(one_hot(game.trump if game.trump is not None else 5, 6))

    # 4. highest bid rank per player, relative, normalized
    bid_rank = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for r in range(NUM_PLAYERS):
        actual = (viewer - r) % NUM_PLAYERS
        bid_rank[r] = game.bid_history[actual][0] / 10.0
    blocks.append(bid_rank)

    # 5. highest bid suit per player, relative (index 5 = hasn't bid yet)
    for r in range(NUM_PLAYERS):
        actual = (viewer - r) % NUM_PLAYERS
        value, suit = game.bid_history[actual]
        blocks.append(one_hot(suit if value != 0 else 5, 6))

    # 6. misere / open misere flags
    blocks.append(np.array([float(game.misere), float(game.open_misere)], dtype=np.float32))

    # 7. passed[], relative
    passed_rel = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for r in range(NUM_PLAYERS):
        actual = (viewer - r) % NUM_PLAYERS
        passed_rel[r] = float(game.passed[actual])
    blocks.append(passed_rel)

    # 8. current highest bidder, relative (index 4 = no bid yet)
    bidder_idx = 4 if game.bet_winner is None else rel_seat(viewer, game.bet_winner)
    blocks.append(one_hot(bidder_idx, 5))

    # 9. dealer/start seat, relative
    blocks.append(one_hot(rel_seat(viewer, game.start_player), NUM_PLAYERS))

    # 10. cards on the table this trick, per relative seat (index NUM_CARDS = none played)
    table = dict(game.current_trick)
    for r in range(NUM_PLAYERS):
        actual = (viewer - r) % NUM_PLAYERS
        card = table.get(actual)
        blocks.append(one_hot(card if card is not None else NUM_CARDS, NUM_CARDS + 1))

    # 11. card history: for every card, who played it (relative seat, index 4 = not played
    # yet) and which trick it was played in (normalized, 0 = not played yet)
    card_history = np.zeros(NUM_CARDS * 6, dtype=np.float32)
    for c in range(NUM_CARDS):
        base = c * 6
        entry = game.card_history.get(c)
        if entry is None:
            card_history[base + 4] = 1.0
        else:
            player, trick = entry
            card_history[base + rel_seat(viewer, player)] = 1.0
            card_history[base + 5] = (trick + 1) / TRICKS_PER_HAND
    blocks.append(card_history)

    # 12. revealed hand: the open-misere bidder's hand is public once play starts
    revealed = np.zeros(NUM_CARDS, dtype=np.float32)
    if game.open_misere and game.phase == Phase.PLAY and game.bet_winner is not None:
        for c in game.hands[game.bet_winner]:
            revealed[c] = 1.0
    blocks.append(revealed)

    # 13. own/opponent team tricks won this hand, normalized
    own_team = viewer % 2
    own_tricks = sum(game.tricks_won[s] for s in range(NUM_PLAYERS) if s % 2 == own_team)
    opp_tricks = sum(game.tricks_won[s] for s in range(NUM_PLAYERS) if s % 2 != own_team)
    blocks.append(np.array([own_tricks / TRICKS_PER_HAND, opp_tricks / TRICKS_PER_HAND], dtype=np.float32))

    return np.concatenate(blocks)
