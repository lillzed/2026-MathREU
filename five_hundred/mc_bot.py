import copy
import random

from . import cards
from .constants import NUM_PLAYERS, Phase
from .game import FiveHundredGame
from .heuristic_play import heuristic_play_action

"""
Perfect-information Monte Carlo (PIMC) card-play bot: samples several
plausible full deals consistent with public information (own hand, cards
already played, revealed open-misere hands, and suit voids inferred from
revoked follows -- see game.py's void_suits), plays the rest of the hand out
in each sampled world via heuristic_play.py, and picks whichever legal card
scored best on average. This reasons about which specific cards are likely
where, rather than following heuristic_play.py's fixed rules blind to the
actual deal, so it's meant to replace it as the primary training opponent.
"""


def _redeal_unseen(game: FiveHundredGame, viewer: int, rng: random.Random) -> list[set[int]]:
    """
    Sample one plausible full deal from `viewer`'s point of view.

    `viewer`'s own hand, all played cards, and any publicly revealed hand
    (an open misere bidder's, once play starts) stay fixed and accurate.
    Every other active seat's hand is a random deal from the remaining
    unseen cards, sized to match that seat's true (public) card count and
    respecting any suit voids already inferred for them this hand. A misere
    hand's sitting-out seat is left untouched and excluded from the unseen
    pool entirely -- its true cards are never seen by anyone and never
    played, so redealing it would only leak information for free.

    Output (list[set[int]]): a full set of 4 hands, indexed by seat.
    """
    hands = [set(game.hands[s]) for s in range(NUM_PLAYERS)]

    revealed_seats = {viewer}
    if game.open_misere and game.phase == Phase.PLAY and game.bet_winner is not None:
        revealed_seats.add(game.bet_winner)

    redeal_seats = [
        s for s in range(NUM_PLAYERS)
        if s not in revealed_seats and s != game.sitting_out_seat
    ]
    if not redeal_seats:
        return hands

    seen = set(game.card_history.keys())
    for s in revealed_seats:
        seen |= hands[s]
    unseen = [c for c in range(cards.CARD_START_INDEX, cards.JOKER_INDEX + 1) if c not in seen]
    rng.shuffle(unseen)

    needs = {s: len(hands[s]) for s in redeal_seats}
    rng.shuffle(redeal_seats)
    pool = list(unseen)
    for s in redeal_seats:
        voids = game.void_suits[s]
        need = needs[s]
        valid = [c for c in pool if cards.effective_suit(c, game.trump) not in voids]
        chosen = valid[:need]
        if len(chosen) < need:
            chosen += [c for c in pool if c not in chosen][: need - len(chosen)]
        hands[s] = set(chosen)
        pool = [c for c in pool if c not in chosen]

    return hands


def _rollout_value(world: FiveHundredGame, deciding_team: int, first_action: int) -> float:
    """
    Play `first_action` for the current player in a determinized world, then
    finish the hand with heuristic_play.py driving every remaining seat, and
    return the resulting score delta for `deciding_team`.
    """
    sim = copy.deepcopy(world)
    result = sim.step(first_action)
    while not result.hand_completed:
        result = sim.step(heuristic_play_action(sim))
    return float(result.team_score_deltas[deciding_team])


def choose_card(game: FiveHundredGame, num_determinizations: int, rng: random.Random) -> int:
    """
    Pick a PLAY-phase action for game.current_player by averaging
    num_determinizations sampled worlds. Plain function (rather than only a
    method) so training_env.py can dispatch it to a multiprocessing pool via
    choose_card_worker -- at any determinization count strong enough to
    actually outplay heuristic_play.py, this is far too slow (double-digit
    milliseconds) to run inline for every non-learner seat of every training
    step.
    """
    legal = game.legal_actions()
    if len(legal) == 1:
        return legal[0]

    team = game.current_player % 2
    totals = {c: 0.0 for c in legal}

    for _ in range(num_determinizations):
        world = copy.deepcopy(game)
        world.hands = _redeal_unseen(game, game.current_player, rng)
        for c in legal:
            totals[c] += _rollout_value(world, team, c)

    return max(legal, key=lambda c: totals[c])


def choose_card_worker(task: tuple[FiveHundredGame, int, int]) -> int:
    """
    multiprocessing.Pool entry point: unlike MonteCarloCardPolicy.act(),
    takes an explicit seed per call instead of an instance-held
    random.Random, since reusing one rng across pooled calls would mean
    every task pickled from the same policy object in the same wavefront
    round starts from identical state (see FiveHundredVecEnv._advance_all).
    """
    game, num_determinizations, seed = task
    return choose_card(game, num_determinizations, random.Random(seed))


class MonteCarloCardPolicy:
    """
    Stateless (safe to share across many concurrent envs, like
    HeuristicCardPolicy): every call reasons entirely from the live `game`
    object passed in, never from instance state except the rng stream.
    """

    def __init__(self, num_determinizations: int = 12, seed: int | None = None) -> None:
        self.num_determinizations = num_determinizations
        self._rng = random.Random(seed)

    def act(self, game: FiveHundredGame) -> int:
        return choose_card(game, self.num_determinizations, self._rng)
