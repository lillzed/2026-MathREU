"""
Action space
    0..42    card id - used as the action during DISCARD and PLAY phases
    43       PASS       (bid phase only)
    44       MISERE     (bid phase only)
    45       OPENMISERE (bid phase only)
    46..70   normal bid: value in 6..10, suit in 0..4 (S,C,D,H,N)
    71..74   joker suit choice: suit in 0..3 (S,C,D,H)      (JOKER_SUIT phase only)
"""

from typing import Iterable, Any
import numpy as np
import numpy.typing as npt

from . import cards
from .constants import NUM_PLAYERS, TRICKS_PER_HAND, Phase
from .game import FiveHundredGame

PASS_ACTION = cards.NUM_CARDS          # 43
MISERE_ACTION = PASS_ACTION + 1        # 44
OPENMISERE_ACTION = PASS_ACTION + 2    # 45
BID_ACTION_BASE = PASS_ACTION + 3      # 46
JOKER_SUIT_ACTION_BASE = BID_ACTION_BASE + 25  # 71
ACTION_SPACE_SIZE = JOKER_SUIT_ACTION_BASE + 4  # 75


def bid_action(value: int, suit: int) -> int:
    """
    Find the ID of a bid.

    Input:
    - value (int) : the value of the bid
    - suit (int) : the ID of the suit of the bid

    Output (int):
    - the ID of the bid
    """
    return BID_ACTION_BASE + (value - 6) * 5 + suit


def decode_bid_action(action: int) -> tuple[int, int]:
    """
    Find the value and suit of a bid.

    Input:
    - action (int) : the ID of the bid

    Output (tuple[int, int]):
    - the value of the bid
    - the ID of the suit of the bid
    """
    offset = action - BID_ACTION_BASE
    return 6 + offset // 5, offset % 5


def joker_suit_action(suit: int) -> int:
    """
    Find the ID of a joker action.

    Input:
    - suit (int) : the ID of the joker's suit

    Output (int):
    - the ID of the action corresponding to playing the joker in the suit
    """
    return JOKER_SUIT_ACTION_BASE + suit


def decode_joker_suit_action(action: int) -> int:
    """
    Find the suit of a joker action.

    Input:
    - action : the ID of the action

    Output (int):
    - the ID of the suit in which the joker was played
    """
    return action - JOKER_SUIT_ACTION_BASE


def action_mask(legal_actions: Iterable[int]) -> npt.NDArray[np.int8]:
    """
    Create a binary mask with 1s at legal action indices.

    Input:
    - legal_actions (Iterable[int]) : the IDs of legal actions

    Output (npt.NDArray[np.int8]):
    - a mask, with 0s at indices of illegal actions and 1s at indices of legal actions
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    mask[list(legal_actions)] = 1
    return mask

_PHASES = (Phase.BIDDING, Phase.DISCARD, Phase.JOKER_SUIT, Phase.PLAY, Phase.GAME_OVER)
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASES)}

_BLOCK_SIZES = [
    cards.NUM_CARDS,       # 1. own hand
    len(_PHASES),          # 2. phase
    6,                     # 3. trump (S,C,D,H,N,none)
    5,                     # 4. joker suit (S,C,D,H,none)
    1,                     # 5. highest bid value, normalized
    6,                     # 6. highest bid suit (S,C,D,H,N,none)
    2,                     # 7. misere / open-misere flags
    NUM_PLAYERS,           # 8. passed[], relative
    5,                     # 9. current highest bidder, relative (+none)
    NUM_PLAYERS,           # 10. dealer/start seat, relative
    *([cards.NUM_CARDS + 1] * NUM_PLAYERS),  # 11. cards on table this trick, per relative seat (+none)
    cards.NUM_CARDS,       # 12. cards seen in earlier tricks this hand
    cards.NUM_CARDS,       # 13. revealed hand (open misere bidder, once play starts)
    2,                     # 14. own/opponent team tricks won this hand, normalized
]
OBS_SIZE = sum(_BLOCK_SIZES)


def _rel_seat(actual_seat: int, viewer: int) -> int:
    """
    Find a player's seat relative to a viewer.

    Input:
    - actual_seat (int) : the ID of the player's actual seat
    - viewer (int) : the ID of the viewers seat

    Output (int):
    - the ID of the player's seat relative to the viewer
    """
    return (actual_seat - viewer) % NUM_PLAYERS


def _onehot(index: int | None, size: int) -> npt.NDArray[np.float32]:
    """
    Create a mask with one (or zero) true value.

    Input:
    - index (int) : the index of the true value
    - size (int) : the length of the mask

    Output (npt.NDArray[np.float32]):
    - a numpy array of length 'size' with 0s at all indices except 1.0 at the designated index
    """
    v = np.zeros(size, dtype=np.float32)
    if index is not None:
        v[index] = 1.0
    return v


def encode_observation(game: FiveHundredGame, viewer: int) -> np.ndarray:
    """
    Take a game state and encode into an RL friendly observation.

    Input:
    - game (FiveHundredGame) : a 500 game
    - viewer (int) : the ID of the player

    Output (np.ndarray):
    - a numpy array fully encoding all relevant game data observable by the viewer
    """
    blocks: list[Any] = []

    # 1. own hand
    own_hand = np.zeros(cards.NUM_CARDS, dtype=np.float32)
    for c in game.hands[viewer]:
        own_hand[c] = 1.0
    blocks.append(own_hand)

    # 2. phase
    blocks.append(_onehot(_PHASE_INDEX.get(game.phase), len(_PHASES)))

    # 3. trump
    blocks.append(_onehot(game.trump if game.trump is not None else 5, 6))

    # 4. joker suit (index 4 = not chosen / not applicable)
    blocks.append(_onehot(game.joker_suit if game.joker_suit is not None else 4, 5))

    # 5. highest bid value, normalized
    blocks.append(np.array([game.highest_bet / 10.0], dtype=np.float32))

    # 6. highest bid suit (index 5 = none yet)
    blocks.append(_onehot(game.highest_suit if game.highest_suit is not None else 5, 6))

    # 7. misere / open misere flags
    blocks.append(np.array([float(game.misere), float(game.open_misere)], dtype=np.float32))

    # 8. passed[], relative
    passed_rel = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for r in range(NUM_PLAYERS):
        passed_rel[r] = float(game.passed[(viewer + r) % NUM_PLAYERS])
    blocks.append(passed_rel)

    # 9. current highest bidder, relative (index 4 = no bid yet)
    bidder_idx = 4 if game.bet_winner is None else _rel_seat(game.bet_winner, viewer)
    blocks.append(_onehot(bidder_idx, 5))

    # 10. dealer/start seat, relative
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
