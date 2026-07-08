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
    JOKER_SUIT = auto()
    PLAY = auto()
    GAME_OVER = auto()


def bid_points(value: int, suit: int) -> int:
    """
    Calculate the point value of a bid.

    Input:
    value (int): the value of the bid
    suit (int): the ID of the suit of the bid

    Output (int):
    - the point value of the bid
    """
    return (4 + suit * 2) * 10 + (value - 6) * 100
