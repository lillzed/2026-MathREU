SPADES, CLUBS, DIAMONDS, HEARTS, NO_TRUMPS = 0, 1, 2, 3, 4
SUITS = (SPADES, CLUBS, DIAMONDS, HEARTS)
SUIT_CHARS = {SPADES: "S", CLUBS: "C", DIAMONDS: "D", HEARTS: "H", NO_TRUMPS: "N"}
CHAR_TO_SUIT = {c: s for s, c in SUIT_CHARS.items()}

JACK, QUEEN, KING, ACE = 11, 12, 13, 14
LEFT_BOWER, RIGHT_BOWER, JOKER_VALUE = 15, 16, 17

JOKER = 42
NUM_CARDS = 43

SAME_COLOUR = {(SPADES, CLUBS), (CLUBS, SPADES), (DIAMONDS, HEARTS), (HEARTS, DIAMONDS)}

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
    Find the rank of a card.

    Input:
    - card (int) : the ID of the card

    Output (int):
    - the rank of the card
    """
    return _RANK_OF[card]


def suit_of(card: int) -> int:
    """
    Find the suit of a card.

    Input:
    - card (int) : the ID of the card

    Output (int):
    - the ID of the suit of the card
    """
    return _SUIT_OF[card]


def format_card(card: int) -> str:
    """
    Create a string representation of a card.

    Input:
    - card (int) : the ID of the card

    Output (str):
    - a string representation of the card in the form "{rank}{suit}"
    """
    if card == JOKER:
        return "JOKER"
    rank, suit = _RANK_OF[card], _SUIT_OF[card]
    return f"{_RANK_CHARS.get(rank, str(rank))}{SUIT_CHARS[suit]}"


def effective_card(card: int, trump: int, joker_suit: int) -> tuple[int, int]:
    """
    Find the effective rank and suit of a card.

    Input:
    - card (int) : the ID of the card
    - trump (int) : the ID of the trump suit
    - joker_suit (int) : the ID of the joker's suit

    Output (tuple[int, int]):
    - an effective rank of the card relative to other cards in the effective suit
    - the ID of the effective suit of the card
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
    Find the effective suit of a card.

    Input:
    - card (int) : the ID of the card
    - trump (int) : the ID of the trump suit
    - joker_suit (int) : the ID of the joker's suit

    Output:
    - the ID of the effective suit of the card
    """
    return effective_card(card, trump, joker_suit)[1]


def compare_cards(a: int, b: int, trump: int, joker_suit: int) -> int:
    """
    Compute which of two cards is superior.

    Input:
    - a (int) : the ID of a card
    - b (int) : the ID of a card
    - trump (int) : the ID of the trump suit
    - joker_suit (int) : the ID of the joker_suit

    Output (int):g
    - 0 if 'a' and 'b' are the same
    - 1 if 'a' is superior to 'b'
    - -1 otherwise
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