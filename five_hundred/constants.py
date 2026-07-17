from enum import Enum, auto

NUM_PLAYERS = 4
TRICKS_PER_HAND = 10
WIN_POINTS = 500

MISERE_POINTS = 250
OPEN_MISERE_POINTS = 500

DEFAULT_MAX_HANDS = 200


class Phase(Enum):
    BIDDING = auto()
    DISCARD = auto()
    PLAY = auto()
    GAME_OVER = auto()


def bid_points(value: int, suit: int) -> int:
    """
    Calculates the point value of a standard bid from the value and suit.
    """
    return (4 + suit * 2) * 10 + (value - 6) * 100