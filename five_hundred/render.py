from . import cards
from . import encoding as enc
from .constants import Phase
from .game import FiveHundredGame, StepResult

# cards.STRING_DICT covers the four real suits plus face ranks (for format_card),
# but bids can also be NO_SUIT ("no trumps"), which isn't a printable card suit.
_SUIT_CHARS = {
    cards.SPADES: "S", cards.CLUBS: "C", cards.DIAMONDS: "D",
    cards.HEARTS: "H", cards.NO_SUIT: "N",
}


def _bid_description(action: int) -> str:
    """
    Describe a bidding-phase action in human-readable form.

    Input:
    - action (int) : the ID of a bid action

    Output (str):
    - a description of the bid, e.g. "passed" or "bid 8H"
    """
    value, suit, misere = enc.decode_bid_action(action)
    if value == 0:
        return "passed"
    if misere:
        return "bid OPEN MISERE" if value == 10 else "bid MISERE"
    return f"bid {value}{_SUIT_CHARS[suit]}"


def describe_action(phase_before: Phase, seat: int, action: int) -> str:
    """
    Describe the action about to be applied, using the phase the game was in
    *before* the step (so a play action still reads as "played", not whatever
    phase the game moved to afterwards).

    Input:
    - phase_before (Phase) : the game's phase prior to applying the action
    - seat (int) : the ID of the acting player
    - action (int) : the ID of the action

    Output (str):
    - a human-readable description of the action
    """
    if phase_before == Phase.BIDDING:
        return f"Player {seat} {_bid_description(action)}"
    if phase_before == Phase.DISCARD:
        return f"Player {seat} discarded {cards.format_card(action)}"
    if phase_before == Phase.PLAY:
        return f"Player {seat} played {cards.format_card(action)}"
    return f"Player {seat} took action {action}"


def describe_new_hand(game: FiveHundredGame) -> str:
    """
    Describe the start of a new hand.

    Input:
    - game (FiveHundredGame) : a 500 game

    Output (str):
    - a banner announcing the hand number and dealer/first bidder
    """
    return f"\n=== Hand {game.hand_number} (dealer/first bidder: Player {game.start_player}) ==="


def describe_redeal() -> str:
    """
    Describe a redeal (every player passed without a bid).

    Input (None)

    Output (str):
    - a message announcing the redeal
    """
    return "  All players passed without a bid - reshuffling and redealing."


def describe_trick_result(result: StepResult) -> str:
    """
    Describe the outcome of a completed trick.

    Input:
    - result (StepResult) : the step result from the play that completed the trick

    Output (str):
    - a message naming the trick winner and winning card
    """
    return f"  -> Player {result.trick_winner} wins the trick with {cards.format_card(result.winning_card)}"


def _contract_description(summary: dict) -> str:
    """
    Describe the contract a hand was played to.

    Input:
    - summary (dict) : a StepResult.hand_summary dict

    Output (str):
    - the contract, e.g. "MISERE", "OPEN MISERE", or "8H"
    """
    if summary["misere"]:
        return "OPEN MISERE" if summary["open_misere"] else "MISERE"
    return f"{summary['highest_bet']}{_SUIT_CHARS[summary['highest_suit']]}"


def describe_hand_result(result: StepResult, game: FiveHundredGame) -> str:
    """
    Describe the outcome of a completed hand.

    Input:
    - result (StepResult) : the step result from the play that completed the hand
    - game (FiveHundredGame) : the game, for current team scores

    Output (str):
    - a message describing the contract, outcome, and scores after the hand
    """
    s = result.hand_summary
    outcome = "WON" if s["success"] else "LOST"
    sign = "+" if s["success"] else "-"
    return (
        f"  Player {s['bidder']} bid {_contract_description(s)}, "
        f"betting team took {s['bidding_team_tricks']} trick(s) -> {outcome} ({sign}{s['points']} points)\n"
        f"  Scores after this hand: Team 0 = {game.team_scores[0]}, Team 1 = {game.team_scores[1]}"
    )


def describe_match_over(game: FiveHundredGame) -> str:
    """
    Describe the end of a match.

    Input:
    - game (FiveHundredGame) : the completed game

    Output (str):
    - a banner announcing the winning team and final scores
    """
    winning_team = 0 if game.team_scores[0] > game.team_scores[1] else 1
    return (
        f"\n*** MATCH OVER - Team {winning_team} wins! "
        f"Final scores: Team 0 = {game.team_scores[0]}, Team 1 = {game.team_scores[1]} ***"
    )
