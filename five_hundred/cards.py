"""Card representation and comparison rules for this project's 500 deck.

Cards are plain ints 0..42 (43-card deck). This mirrors 500/cards.c
create_deck exactly: ranks 5-14 ("5".."A") in all 4 suits (40 cards), plus
only the *red* 4s (4D, 4H - no black 4s), plus the Joker.

Card id layout (not required to match the C server's internal order,
only its composition - id assignment here is just a convenient encoding):
    0..39  -> rank in [5..14], suit in [SPADES,CLUBS,DIAMONDS,HEARTS],
              id = (rank - 5) * 4 + suit
    40     -> 4 of Diamonds
    41     -> 4 of Hearts
    42     -> Joker
"""

SPADES, CLUBS, DIAMONDS, HEARTS, NO_TRUMPS = 0, 1, 2, 3, 4
SUITS = (SPADES, CLUBS, DIAMONDS, HEARTS)
SUIT_CHARS = {SPADES: "S", CLUBS: "C", DIAMONDS: "D", HEARTS: "H", NO_TRUMPS: "N"}
CHAR_TO_SUIT = {c: s for s, c in SUIT_CHARS.items()}

JACK, QUEEN, KING, ACE = 11, 12, 13, 14
LEFT_BOWER, RIGHT_BOWER, JOKER_VALUE = 15, 16, 17

JOKER = 42
NUM_CARDS = 43

# same-colour suit pairs, used for the bower rule (jack of trump's colour
# becomes the "left bower" and counts as trump)
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


def rank_of(card):
    return _RANK_OF[card]


def suit_of(card):
    return _SUIT_OF[card]


def format_card(card):
    """Human-readable token, e.g. "8H", "10S", "JOKER" - for debugging/render only."""
    if card == JOKER:
        return "JOKER"
    rank, suit = _RANK_OF[card], _SUIT_OF[card]
    return f"{_RANK_CHARS.get(rank, str(rank))}{SUIT_CHARS[suit]}"


def effective_card(card, trump, joker_suit):
    """Returns (effective_value, effective_suit) for trick comparison and
    follow-suit checks. Mirrors cards.c handle_bower, plus the joker-suit
    substitution server.c applies (get_valid_card_from_player) before
    calling correct_suit_player/compare_cards: the joker always counts as
    the trump suit, or joker_suit when trump is no-trumps."""
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


def effective_suit(card, trump, joker_suit):
    return effective_card(card, trump, joker_suit)[1]


def compare_cards(a, b, trump, joker_suit):
    """Returns 1 if card a beats card b, else -1 (0 only if a == b).
    b is the card currently winning the trick. Mirrors cards.c compare_cards."""
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
