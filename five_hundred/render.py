from . import cards
from . import encoding as enc
from .constants import Phase


def _bid_description(action):
    if action == enc.PASS_ACTION:
        return "passed"
    if action == enc.MISERE_ACTION:
        return "bid MISERE"
    if action == enc.OPENMISERE_ACTION:
        return "bid OPEN MISERE"
    value, suit = enc.decode_bid_action(action)
    return f"bid {value}{cards.SUIT_CHARS[suit]}"


def describe_action(phase_before, seat, action):
    """Describes the action about to be applied, using the phase the game
    was in *before* the step (so a play action still reads as "played",
    not whatever phase the game moved to afterwards)."""
    if phase_before == Phase.BIDDING:
        return f"Player {seat} {_bid_description(action)}"
    if phase_before == Phase.DISCARD:
        return f"Player {seat} discarded {cards.format_card(action)}"
    if phase_before == Phase.JOKER_SUIT:
        suit = enc.decode_joker_suit_action(action)
        return f"Player {seat} chose {cards.SUIT_CHARS[suit]} as the joker's suit"
    if phase_before == Phase.PLAY:
        return f"Player {seat} played {cards.format_card(action)}"
    return f"Player {seat} took action {action}"


def describe_new_hand(game):
    return f"\n=== Hand {game.hand_number} (dealer/first bidder: Player {game.start_player}) ==="


def describe_redeal():
    return "  All players passed without a bid - reshuffling and redealing."


def describe_trick_result(result):
    return f"  -> Player {result.trick_winner} wins the trick with {cards.format_card(result.winning_card)}"


def _contract_description(summary):
    if summary["misere"]:
        return "OPEN MISERE" if summary["open_misere"] else "MISERE"
    return f"{summary['highest_bet']}{cards.SUIT_CHARS[summary['highest_suit']]}"


def describe_hand_result(result, game):
    s = result.hand_summary
    outcome = "WON" if s["success"] else "LOST"
    sign = "+" if s["success"] else "-"
    return (
        f"  Player {s['bidder']} bid {_contract_description(s)}, "
        f"betting team took {s['bidding_team_tricks']} trick(s) -> {outcome} ({sign}{s['points']} points)\n"
        f"  Scores after this hand: Team 0 = {game.team_scores[0]}, Team 1 = {game.team_scores[1]}"
    )


def describe_match_over(game):
    winning_team = 0 if game.team_scores[0] > game.team_scores[1] else 1
    return (
        f"\n*** MATCH OVER - Team {winning_team} wins! "
        f"Final scores: Team 0 = {game.team_scores[0]}, Team 1 = {game.team_scores[1]} ***"
    )
