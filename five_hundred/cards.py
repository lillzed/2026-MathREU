SPADES, CLUBS, DIAMONDS, HEARTS, NO_TRUMPS = 0, 1, 2, 3, 4
SUITS = (SPADES, CLUBS, DIAMONDS, HEARTS)
SUIT_CHARS = {SPADES: "S", CLUBS: "C", DIAMONDS: "D", HEARTS: "H", NO_TRUMPS: "N"}
CHAR_TO_SUIT = {c: s for s, c in SUIT_CHARS.items()}

JACK, QUEEN, KING, ACE = 11, 12, 13, 14
LEFT_BOWER, RIGHT_BOWER, JOKER_VALUE = 15, 16, 17

JOKER = 42
NUM_CARDS = 43

SAME_COLOUR = {(SPADES, CLUBS), (CLUBS, SPADES), (DIAMONDS, HEARTS), (HEARTS, DIAMONDS)}

# For each card, assign it an ID from 0-42 and fill the dictionaries _RANK_OF and _SUIT_OF by [ID: rank/suit].
_RANK_OF = {}
_SUIT_OF = {}
for _rank in range(5, 15):
    for _suit in SUITS:
        _card = (_rank - 5) * 4 + _suit
        _RANK_OF[_card] = _rank
        _SUIT_OF[_card] = _suit
_RANK_OF[40], _SUIT_OF[40] = 4, DIAMONDS
_RANK_OF[41], _SUIT_OF[41] = 4, HEARTS
_RANK_OF[JOKER], _SUIT_OF[JOKER] = JOKER_VALUE, NO_TRUMPS

_RANK_CHARS = {JACK: "J", QUEEN: "Q", KING: "K", ACE: "A"}


def rank_of(card: int) -> int:
    """
    Given the ID of a card, returns the rank of the card.
    """
    return _RANK_OF[card]


def suit_of(card: int) -> int:
    """
    Given the ID of a card, returns the ID of the suit of the card.
    """
    return _SUIT_OF[card]


def format_card(card: int) -> str:
    """
    Given the ID of a card, returns a string representation of the card.
    """
    if card == JOKER:
        return "JOKER"
    rank, suit = _RANK_OF[card], _SUIT_OF[card]
    return f"{_RANK_CHARS.get(rank, str(rank))}{SUIT_CHARS[suit]}"


def effective_card(card: int, trump: int, joker_suit: int) -> tuple[int, int]:
    """
    Given the ID of a card, trump suit, and joker_suit, returns the tuple of the effective rank and suit of the card.
    """
    if card == JOKER:
        suit = joker_suit if trump == NO_TRUMPS else trump
        return JOKER_VALUE, suit

    value, suit = _RANK_OF[card], _SUIT_OF[card]
    if value != JACK:
        return value, suit
    if suit == trump:
        return RIGHT_BOWER, trump
    if (suit, trump) in SAME_COLOUR:
        return LEFT_BOWER, trump
    return value, suit


def effective_suit(card: int, trump: int, joker_suit: int) -> int:
    """
    Given the ID of a card, trump suit, and joker_suit, returns the effective suit of the card.
    """
    return effective_card(card, trump, joker_suit)[1]


def compare_cards(a: int, b: int, trump: int, joker_suit: int) -> int:
    """
    Given the ID's of two cards, the trump suit, and the joker_suit, returns 0 if the cards are equal, 1 if a beats b, and -1 if b beats a
    """
    if a == b:
        return 0
    va, sa = effective_card(a, trump, joker_suit)
    vb, sb = effective_card(b, trump, joker_suit)
    if sa == trump and sb != trump:
        return 1
    if sb == trump and sa != trump:
        return -1
    if sa == sb:
        return 1 if va > vb else -1
    return -1