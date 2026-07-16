import random

from .cards import (
    effective_card,
    QUEEN, KING, ACE, LEFT_BOWER, RIGHT_BOWER, JOKER_VALUE,
)

SPADE = 0
CLUB = 1
DIAMOND = 2
HEART = 3
NT = 4
MISERE = 5
SUIT_KEY = {0: "SPADE", 1: "CLUB", 2: "DIAMOND",
            3: "HEART", 4: "NT", 5: "MISERE"}

TRUMP_RANK_BONUS = {
    JOKER_VALUE: 1.0,
    RIGHT_BOWER: 1.0,
    LEFT_BOWER: 0.95,
    ACE: 0.9,
    KING: 0.75,
    QUEEN: 0.6,
    10: 0.48,
    9: 0.4,
    8: 0.32,
    7: 0.26,
    6: 0.2,
    5: 0.15,
    4: 0.1
}


def _make_suit_bid(cards: list[int], suit: int) -> float:
    """
    Estimate expected tricks if `suit` is trump.
    """
    effective = [effective_card(card, suit, NT) for card in cards]
    trump = sorted((r for r, s in effective if s == suit), reverse=True)
    offsuit: dict[int, list[int]] = {}
    for r, s in effective:
        if s != suit:
            offsuit.setdefault(s, []).append(r)

    tricks = sum(TRUMP_RANK_BONUS[r] for r in trump)
    if len(trump) > 1:
        # each trump beyond the first further draws out opponents' trumps,
        # promoting the rest of the suit
        tricks += (len(trump) - 1) * 0.45

    for s in range(4):
        if s == suit:
            continue
        ranks = sorted(offsuit.get(s, []), reverse=True)
        if not ranks:
            tricks += 0.85                         # void: ruff whenever led
            continue
        if ranks[0] >= 14:                         # ace
            tricks += 1.1
        elif ranks[0] >= 13 and len(ranks) > 1:     # guarded king
            tricks += 0.6
        if len(ranks) == 1:                         # singleton: ruff after one lead
            tricks += 0.55

    return min(tricks, 10.0)   # a hand only has 10 tricks to win


def _make_nt_bid(cards: list[int]) -> float:
    """
    Estimate expected tricks with no trump suit (no bower promotion).
    """
    effective = [effective_card(card, NT, NT) for card in cards]
    by_suit: dict[int, list[int]] = {}
    for r, s in effective:
        by_suit.setdefault(s, []).append(r)

    tricks = 0.0
    for ranks in by_suit.values():
        ranks.sort(reverse=True)
        has_ace = ranks[0] >= 14
        has_king = 13 in ranks
        guarded = len(ranks) > 1

        if has_ace:
            tricks += 1.1
        if has_king and guarded:
            tricks += 0.65
        elif has_king:
            tricks += 0.3
        if len(ranks) >= 5 and not has_ace:
            tricks -= 0.3

    return min(max(0.0, tricks), 10.0)   # a hand only has 10 tricks to win


def _misere_score(cards: list[int]) -> float:
    """
    Estimate a misere bid's value on the same ~0-10 scale used for the
    trick-count scores, so it can be compared directly against them.
    """
    effective = [effective_card(card, NT, NT) for card in cards]
    by_suit: dict[int, list[int]] = {}
    for r, s in effective:
        by_suit.setdefault(s, []).append(r)

    danger = 0.0
    for ranks in by_suit.values():
        ranks.sort()
        top = ranks[-1]
        cover = len(ranks) - 1
        card_danger = max(0.0, (top - 8) / 2)
        card_danger = max(0.0, card_danger - 0.3 * cover)
        danger += card_danger

    for s in range(4):
        held = by_suit.get(s, [])
        if not held:
            danger += 0.5                          # void
        elif len(held) == 1 and held[0] > 9:
            danger += 0.5                           # unprotected singleton

    safety = max(0.0, 10.0 - danger)

    if safety >= 8.5:
        return 7.5 + (safety - 8.5) * (2.5 / 1.5)   # near-flawless hand -> up to 10
    return safety * 0.5                              # too risky to compete with a real bid


def make_bid(cards: list[int]) -> tuple[int, int]:
    """
    Based on a list of cards, make a bid.

    Input:
    - cards (list[int]) : the ID's of the player's cards

    Output:
    - (contract name, expected-trick score). Score is on a common ~0-10
      scale across suits/NT/misere so callers can threshold on it (e.g.
      pass below 6)
    """
    scores = [0.0] * 6
    for suit in range(4):
        scores[suit] = _make_suit_bid(cards, suit)
    scores[NT] = _make_nt_bid(cards)
    scores[MISERE] = _misere_score(cards)

    best = scores.index(max(scores))
    return (best, round(scores[best]))


#___________________Kitty Heuristic_________________#

def _discard_kitty_suit(cards: list[int], trump: int) -> tuple[int, int, int]:
    """
    Starting with the list of cards, removes trumps, then aces, then protected kings,
    while checking that the number of candidate cards is greater than or equal to 3. Then, short
    suits whatever suit has cards with the lowest sum.
    """
    
    effective_cards = [effective_card(card, trump, trump) for card in cards]
    
    candidates = [card for card in effective_cards if card[1] != trump else 0]
    if len(candidates) < 3:

     




def _discard_kitty_misere(cards: list[int], bid: int) -> tuple[int, int, int]:
    raise NotImplementedError

def _discard_kitt_nt(cards: list[int], bid: int) -> tuple[int, int, int]:
    raise NotImplementedError

def discard_kitty(cards: list[int], bid: int) -> tuple[int, int, int]:
    raise NotImplementedError