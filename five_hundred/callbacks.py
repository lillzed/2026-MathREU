import copy

from stable_baselines3.common.callbacks import BaseCallback  # type: ignore


class OpponentPoolCallback(BaseCallback):
    """
    Periodically freezes a copy of the current policy into the shared
    OpponentPool (five_hundred/opponent_pool.py) used by FiveHundredVecEnv
    to drive the non-learner seats during self-play.

    Input:
    - update_freq (int) : push a new snapshot every this many calls to
      _on_step (i.e. every update_freq * num_envs environment timesteps,
      matching how CheckpointCallback's save_freq is scaled in train.py)
    """

    def __init__(self, update_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.update_freq = update_freq

    def _on_training_start(self) -> None:
        self._push_snapshot()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self._push_snapshot()
        return True

    def _push_snapshot(self) -> None:
        snapshot = copy.deepcopy(self.model.policy).to("cpu")
        self.training_env.env_method("add_opponent_snapshot", snapshot)
