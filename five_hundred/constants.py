"""Game-wide constants for this project's variant of 500.

Values here are pinned to the reference C implementation in 500/server.c
and 500/cards.c (server.h: NUM_ROUNDS=10, NUM_PLAYERS=4, WIN_POINTS=500).
"""

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


def bid_points(value, suit):
    """Points for a normal (non-misere) contract. Mirrors server.c
    get_points_from_bet: (4 + suit*2)*10 + (bid-6)*100, i.e. S=40,C=60,
    D=80,H=100,NT=120 base plus 100 per trick bid above 6."""
    return (4 + suit * 2) * 10 + (value - 6) * 100
