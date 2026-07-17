SPADES, CLUBS, DIAMONDS, HEARTS, NO_SUIT = 0, 1, 2, 3, 4
COLOR_DICT = {SPADES: CLUBS, CLUBS: SPADES, DIAMONDS: HEARTS, HEARTS: DIAMONDS}

CARD_START_INDEX = 0
JOKER_INDEX = 42
JACK, QUEEN, KING, ACE, JOKER = 11, 12, 13, 14, 15
NUM_CARDS = 43

ID_CARD = {CARD_START_INDEX: (4, DIAMONDS), CARD_START_INDEX + 1: (4, HEARTS)}
for i in range (CARD_START_INDEX + 2, CARD_START_INDEX + NUM_CARDS):
    ID_CARD[i] = (((18 + (i - CARD_START_INDEX)) // 4), i % 4)
ID_CARD[JOKER_INDEX] = (15, NO_SUIT)
CARD_ID = {c: i for i, c in ID_CARD.items()}

STRING_DICT = {SPADES: "S", CLUBS: "C", DIAMONDS: "D", HEARTS: "H",
               JACK: "J", QUEEN: "Q", KING: "K", ACE: "A"}

def rank_of(card: int) -> int:
    """Returns the rank of a card (4 - 15) given the cards ID."""
    return ID_CARD[card][0]


def suit_of(card: int) -> int:
    """Returns the ID of the suit of a card (0 - 4) given the cards ID."""
    return ID_CARD[card][1]


def id_card(rank: int, suit: int) -> int:
    """Returns the ID of a card given the cards rank and suit."""
    return CARD_ID[(rank, suit)]


def format_card(card: int) -> str:
    """Formats a card into a string representation given the cards ID."""
    if card == JOKER_INDEX:
        return "JOKER"
    
    suit = STRING_DICT[suit_of(card)]
    
    rank = rank_of(card)
    if rank >= JACK:
        rank = STRING_DICT[rank]
    
    return str(rank) + suit


def effective_card(card: int, trump: int) -> tuple[int, int]:
    """
    Returns (order, suit), where 'order' is a number strictly used for comparison
    within the same suit and 'suit' is the effective suit of a card.
    """
    rank = rank_of(card)
    suit = suit_of(card)

    if rank == JACK and suit == trump:
        return (JOKER_INDEX + 2, trump)
    elif rank == JACK and trump != NO_SUIT and suit == COLOR_DICT[trump]:
        return (JOKER_INDEX + 1, trump)
    elif rank == JOKER:
        return (JOKER_INDEX + 3, trump)
    
    return (rank, suit)


def effective_suit(card: int, trump: int) -> int:
    """Returns the effective suit of a card given the ID of the card and the trump suit."""
    return effective_card(card, trump)[1]


def compare_cards(card1: int, card2: int, trump: int, lead: int = NO_SUIT) -> int:
    """
    Returns 1 if card1 beats card2, -1 if card2 beats card1, and 0 if more info is needed,
    given the IDs of both cards, the trump suit, and optionally the card led.
    """
    rank1, suit1 = effective_card(card1, trump)
    rank2, suit2 = effective_card(card2, trump)

    if suit1 == suit2:
        return 1 if rank1 > rank2 else -1
    elif suit1 == trump or suit2 == trump:
        return 1 if suit1 == trump else -1
    elif suit1 == lead or suit2 == lead:
        return 1 if suit1 == lead else -1
    return 0