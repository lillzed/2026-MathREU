"""
500 Card Game - Rules Engine
============================
Pure-Python port of just the rule logic needed to compute legal actions
from the game's text log: bid legality (server.c valid_bet / MISERE /
OPENMISERE handling) and follow-suit legality for card play
(cards.c correct_suit_player / handle_bower). Card tokens use the same
string format the game server prints (e.g. "8H", "10S", "JOKER").

No dependency on the C server at runtime - this only needs to reproduce
the small slice of game rules that determine what actions were legal at
a given decision point, so decision-level data can be reconstructed from
collected game logs.
"""

SUIT_ORDER = ['S', 'C', 'D', 'H', 'N']  # spades < clubs < diamonds < hearts < no-trumps
SUIT_RANK = {s: i for i, s in enumerate(SUIT_ORDER)}
SAME_COLOUR = {('S', 'C'), ('C', 'S'), ('D', 'H'), ('H', 'D')}

VALID_VALUE_CHARS = set('456789') | {'J', 'Q', 'K', 'A'}


def is_valid_card(token):
    """
    True for a real card token. bot.c's get_lowest_non_trump_card has a
    documented edge case (bot.c:117-120): if the kitty winner's 13-card
    hand has fewer than 3 non-trump cards, it returns a sentinel "0S"
    card that gets logged via "Discarded: 0S" without actually removing
    a real card from the bot's hand. That sentinel (return_card's
    invalid-card fallback, cards.c:100-102) isn't a real card and can't
    be reasoned about by this module.
    """
    if token == "JOKER":
        return True
    if token.startswith('10'):
        return len(token) == 3 and token[2] in 'SCDH'
    return len(token) == 2 and token[0] in VALID_VALUE_CHARS and token[1] in 'SCDH'


def parse_card(token):
    """Returns (value, suit) for a card token. Joker has no real suit."""
    if token == "JOKER":
        return (17, 'N')

    if token.startswith('10'):
        return (10, token[2])

    value_char, suit_char = token[0], token[1]
    if value_char in '456789':
        value = int(value_char)
    else:
        value = {'J': 11, 'Q': 12, 'K': 13, 'A': 14}[value_char]

    return (value, suit_char)


def card_sort_key(token):
    value, suit = parse_card(token)
    return (SUIT_RANK.get(suit, 5), value)


def handle_bower(value, suit, trump):
    """Mirrors cards.c handle_bower: swaps a jack into a left/right bower."""
    if value != 11:  # only jacks can be bowers
        return value, suit

    if suit == trump:
        return 16, trump  # right bower

    if trump in ('S', 'C', 'D', 'H') and (suit, trump) in SAME_COLOUR:
        return 15, trump  # left bower

    return value, suit


def effective_suit(token, trump, joker_suit):
    """
    The suit a card counts as for following-suit purposes. Mirrors how
    server.c translates the joker's suit before calling correct_suit_player
    (get_valid_card_from_player, server.c:1054-1061): the joker always
    counts as the trump suit, or the chosen joker_suit in a no-trumps
    contract.
    """
    if token == "JOKER":
        return joker_suit if trump == 'N' else trump

    value, suit = parse_card(token)
    _, eff = handle_bower(value, suit, trump)
    return eff


def legal_play_actions(hand, trump, lead, joker_suit):
    """
    Mirrors correct_suit_player (cards.c:438): if the player holds any
    card matching the led suit they must play one of those, otherwise any
    card is legal. lead=None means this player is leading the trick.
    """
    if lead is None:
        return list(hand)

    matching = [c for c in hand if effective_suit(c, trump, joker_suit) == lead]
    return matching if matching else list(hand)


def legal_discard_actions(remaining_pool):
    """During the kitty discard phase any remaining card may be discarded."""
    return list(remaining_pool)


def legal_bid_actions(highest_bet, highest_suit, misere_taken, open_taken):
    """
    Mirrors valid_bet (server.c:861) plus the MISERE (server.c:949) and
    OPENMISERE (server.c:970) conditions. Does not depend on the bidder's
    hand - matches the real betting rules.
    """
    actions = ["PASS"]
    hb = highest_bet or 0
    hs_rank = SUIT_RANK.get(highest_suit, -1)

    for value in range(6, 11):
        for suit in SUIT_ORDER:
            if value > hb or (value == hb and SUIT_RANK[suit] > hs_rank):
                actions.append(f"{value}{suit}")

    if hb == 7 and not misere_taken:
        actions.append("MISERE")

    if (hb < 10 or (hb == 10 and hs_rank <= SUIT_RANK['D'])) and not open_taken:
        actions.append("OPENMISERE")

    return actions


def normalize_bid_token(raw):
    """Normalizes a bid token parsed from the log to the wire-protocol form
    bots actually send (server.c get_valid_bet_from_player): PASS,
    MISERE, OPENMISERE, or "<value><suit>" e.g. "8H"."""
    if raw is None:
        return "PASS"
    low = raw.lower()
    if low == "misere":
        return "MISERE"
    if low == "openmisere":
        return "OPENMISERE"
    return raw.upper()
