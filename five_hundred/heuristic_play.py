from . import cards
from .game import FiveHundredGame

"""
Rule-based card play, used as a fixed opponent in the self-play pool (see
opponent_pool.py: HeuristicCardPolicy). Not a strong all-around player, just
sound fundamentals: win tricks as cheaply as possible, draw trumps from the
top down, and handle misere's inverted card values correctly on both sides
of the contract. Only ever asked for PLAY-phase actions -- bidding and
discarding are resolved by heuristic_bidding via hybrid_env.py.
"""


def _current_best(game: FiveHundredGame) -> tuple[int, int]:
    """The (player, card) currently winning the trick in progress."""
    best_player, best_card = game.current_trick[0]
    for player, card in game.current_trick[1:]:
        if cards.compare_cards(card, best_card, game.trump, game.lead_suit) == 1:
            best_player, best_card = player, card
    return best_player, best_card


def _lowest(hand: list[int], trump: int) -> int:
    return min(hand, key=lambda c: cards.effective_card(c, trump)[0])


def _highest(hand: list[int], trump: int) -> int:
    return max(hand, key=lambda c: cards.effective_card(c, trump)[0])


def _lead(game: FiveHundredGame, legal: list[int]) -> int:
    """
    Draw trumps from the top down when holding enough of them to make that
    worthwhile; otherwise lead the lowest card in hand.
    """
    if game.trump != cards.NO_SUIT:
        trump_cards = [c for c in legal if cards.effective_suit(c, game.trump) == game.trump]
        if len(trump_cards) >= 2:
            return _highest(trump_cards, game.trump)
    return _lowest(legal, game.trump)


def _follow(game: FiveHundredGame, legal: list[int]) -> int:
    """
    Support a partner who's already winning by playing low; otherwise win as
    cheaply as possible, or duck low if winning isn't possible.
    """
    best_player, best_card = _current_best(game)
    if best_player % 2 == game.current_player % 2:
        return _lowest(legal, game.trump)

    winning = [c for c in legal if cards.compare_cards(c, best_card, game.trump, game.lead_suit) == 1]
    if winning:
        return _lowest(winning, game.trump)
    return _lowest(legal, game.trump)


def _misere_declarer_play(game: FiveHundredGame, legal: list[int]) -> int:
    """
    As misere declarer, spend dangerous high cards early while it's still
    safe (i.e. play the highest card that's still guaranteed to lose), and
    lead from the longest suit to spread risk as evenly as possible.
    """
    if not game.current_trick:
        by_suit: dict[int, list[int]] = {}
        for c in legal:
            by_suit.setdefault(cards.suit_of(c), []).append(c)
        longest_suit = max(by_suit, key=lambda s: len(by_suit[s]))
        return _lowest(by_suit[longest_suit], game.trump)

    _, best_card = _current_best(game)
    safe = [c for c in legal if cards.compare_cards(c, best_card, game.trump, game.lead_suit) != 1]
    if safe:
        return _highest(safe, game.trump)
    return _lowest(legal, game.trump)  # forced to win -- take it as cheaply as possible


def _misere_defender_play(game: FiveHundredGame, legal: list[int]) -> int:
    """
    Defending a misere: keep low cards and force the declarer to win a
    trick. If the declarer is currently winning the trick, duck under with
    the highest card that stays below theirs (sheds a high card without
    letting them off the hook); otherwise play low so the declarer has as
    little room to duck under as possible.
    """
    if game.current_trick:
        best_player, best_card = _current_best(game)
        if best_player == game.bet_winner:
            safe = [c for c in legal if cards.compare_cards(c, best_card, game.trump, game.lead_suit) != 1]
            if safe:
                return _highest(safe, game.trump)
    return _lowest(legal, game.trump)


def heuristic_play_action(game: FiveHundredGame) -> int:
    """
    Choose a PLAY-phase action for game.current_player using simple,
    hand-verifiable card-conservation rules.

    Input:
    - game (FiveHundredGame) : a game currently in Phase.PLAY

    Output (int):
    - the ID of the chosen card to play
    """
    legal = game.legal_actions()
    if game.misere:
        if game.current_player == game.bet_winner:
            return _misere_declarer_play(game, legal)
        return _misere_defender_play(game, legal)
    if not game.current_trick:
        return _lead(game, legal)
    return _follow(game, legal)
