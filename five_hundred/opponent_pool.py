import copy
import random
from typing import Any, Optional

import numpy as np

from .game import FiveHundredGame
from .heuristic_play import heuristic_play_action


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
# uniform share preserves pressure against older strategies.
_RECENT_PROB = 0.5
_RECENT_WINDOW = 20


class OpponentPool:
    """
    Holds frozen copies of the training policy, sampled by FiveHundredVecEnv
    to control the non-learner seats during self-play. Updated periodically
    from train.py via OpponentPoolCallback, so opponents get stronger over
    the course of training without ever training against a policy that's
    still changing underneath them (see five_hundred/callbacks.py).
    """

    def __init__(
        self, max_size: int, seed: Optional[int] = None, heuristic_weight: float = 0.2
    ) -> None:
        self.max_size = max_size
        self.heuristic_weight = heuristic_weight
        self._snapshots: list[Any] = []
        self._rng = random.Random(seed)
        self._fallback = _RandomLegalPolicy()
        self._heuristic = HeuristicCardPolicy()

    def add(self, policy: Any) -> None:
        snapshot = copy.deepcopy(policy).to("cpu")
        snapshot.set_training_mode(False)
        for param in snapshot.parameters():
            param.requires_grad_(False)

        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)

    def sample(self) -> Any:
        if self._rng.random() < self.heuristic_weight:
            return self._heuristic
        if not self._snapshots:
            return self._fallback
        if self._rng.random() < _RECENT_PROB:
            recent = self._snapshots[-_RECENT_WINDOW:]
            return self._rng.choice(recent)
        return self._rng.choice(self._snapshots)
