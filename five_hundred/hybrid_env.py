from pettingzoo.utils import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper

from . import encoding as enc
from . import heuristic_bidding as hb
from .constants import Phase
from .env import env as base_env

# how confident heuristic_bidding's misere score needs to be before declaring
# open misere instead of a regular misere
_OPEN_MISERE_THRESHOLD = 9


def _heuristic_bid_action(game) -> int:
    """
    Turn heuristic_bidding.make_bid()'s hand estimate into a legal bid action
    for the game's current player: bid the estimated capacity if it's still
    achievable given the current auction state, otherwise pass.
    """
    contract, score = hb.make_bid(list(game.hands[game.current_player]))
    if score < 6:
        return enc.bid_action()

    if contract == hb.MISERE:
        value = 10 if score >= _OPEN_MISERE_THRESHOLD else 7
        desired = enc.bid_action(value, misere=True)
    else:
        value = max(6, min(10, score))
        desired = enc.bid_action(value, contract)

    legal = game.legal_actions()
    return desired if desired in legal else enc.bid_action()


def _heuristic_discard_action(game) -> int:
    """
    Ask heuristic_bidding.discard_kitty() for its 3 recommended discards and
    take the first. Re-called fresh each discard-phase turn (on whatever's
    left of the hand), so no cross-step state needs to be tracked.
    """
    if game.misere:
        contract = hb.MISERE
    elif game.highest_suit == enc.NO_SUIT:
        contract = hb.NT
    else:
        contract = game.highest_suit
    hand = list(game.hands[game.current_player])
    return hb.discard_kitty(hand, contract)[0]


def _heuristic_action(game) -> int:
    if game.phase == Phase.BIDDING:
        return _heuristic_bid_action(game)
    return _heuristic_discard_action(game) 


class PlayOnlyEnv(BaseWrapper):
    """
    Wraps a FiveHundredEnv so bidding and discarding are resolved automatically
    by heuristic_bidding, and only PLAY-phase turns are ever exposed through
    step()/agent_iter()/last().
    """

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.env.reset(seed=seed, options=options)
        self._autoplay()

    def step(self, action) -> None:
        self.env.step(action)
        self._autoplay()

    def _autoplay(self) -> None:
        game = self.env.unwrapped._game
        while self.agents:
            agent = self.agent_selection
            if self.terminations[agent] or self.truncations[agent]:
                return
            if game.phase not in (Phase.BIDDING, Phase.DISCARD):
                return
            self.env.step(_heuristic_action(game))


def play_only_env(**kwargs) -> AECEnv:
    """
    Build a fully-wrapped 500 AECEnv where bidding/discard is handled by
    heuristic_bidding, so the caller only ever needs to decide which card to
    play. Forwards kwargs to env.env() (e.g. render_mode, max_hands).
    """
    return PlayOnlyEnv(base_env(**kwargs))
