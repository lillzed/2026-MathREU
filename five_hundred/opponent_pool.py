import copy
import random
from typing import Any, Optional, Protocol

import numpy as np

from .game import FiveHundredGame
from .heuristic_play import heuristic_play_action


class GameStatePolicy(Protocol):
    """
    An opponent that decides from the live game object directly (needs
    hidden state like void inference or hand contents, not just the
    encoded observation/action-mask a trained policy predicts from). See
    FiveHundredVecEnv._advance_to_learner, which dispatches to these via
    act() instead of predict().
    """

    def act(self, game: FiveHundredGame) -> int: ...


class _RandomLegalPolicy:
    """
    Fallback opponent used only before any real snapshot has been added to
    the pool (the very start of training, before OpponentPoolCallback's
    first push): picks a uniformly random legal action.
    """

    def predict(
        self, observation: np.ndarray, action_masks: np.ndarray, deterministic: bool = False
    ) -> tuple[int, None]:
        legal = np.flatnonzero(action_masks)
        return int(np.random.choice(legal)), None


class HeuristicCardPolicy:
    """
    Wraps heuristic_play.py's rule-based card player so it can sit in the
    opponent pool alongside frozen policy snapshots. It needs the live game
    object rather than the encoded observation/action-mask, so
    FiveHundredVecEnv dispatches to act() via an isinstance check instead
    of calling .predict() (see training_env.py's _advance_to_learner).
    Permanent pool member, never evicted.
    """

    def act(self, game: FiveHundredGame) -> int:
        return heuristic_play_action(game)


# sample() draws this fraction of opponents uniformly from the newest
# _RECENT_WINDOW snapshots; the rest come uniformly from the whole pool.
# Mostly-recent opponents keep the gradient signal relevant, while the
# uniform share preserves pressure against older strategies. Only relevant
# when anchor_weight < 1.0 lets some episodes fall through to self-play.
_RECENT_PROB = 0.5
_RECENT_WINDOW = 20


class OpponentPool:
    """
    Sampled by FiveHundredVecEnv to control the non-learner seats each
    episode. By default every episode plays against `anchor` (see
    mc_bot.MonteCarloCardPolicy); anchor_weight < 1.0 lets the remainder
    fall through to frozen self-play snapshots of the training policy
    itself, added periodically by train.py via OpponentPoolCallback (see
    five_hundred/callbacks.py) so self-play, if used at all, never trains
    against a policy that's still changing underneath it.
    """

    def __init__(
        self,
        max_size: int,
        seed: Optional[int] = None,
        anchor_weight: float = 1.0,
        anchor: Optional[GameStatePolicy] = None,
    ) -> None:
        self.max_size = max_size
        self.anchor_weight = anchor_weight
        self._snapshots: list[Any] = []
        self._rng = random.Random(seed)
        self._fallback = _RandomLegalPolicy()
        self._anchor = anchor if anchor is not None else HeuristicCardPolicy()

    def add(self, policy: Any) -> None:
        snapshot = copy.deepcopy(policy).to("cpu")
        snapshot.set_training_mode(False)
        for param in snapshot.parameters():
            param.requires_grad_(False)

        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)

    def sample(self) -> Any:
        if self._rng.random() < self.anchor_weight:
            return self._anchor
        if not self._snapshots:
            return self._fallback
        if self._rng.random() < _RECENT_PROB:
            recent = self._snapshots[-_RECENT_WINDOW:]
            return self._rng.choice(recent)
        return self._rng.choice(self._snapshots)
