import argparse
import os
import re

from sb3_contrib import MaskablePPO #type: ignore
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback #type: ignore
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy #type: ignore
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback #type: ignore
from stable_baselines3.common.vec_env import VecMonitor #type: ignore

from five_hundred.callbacks import OpponentPoolCallback
from five_hundred.opponent_pool import OpponentPool
from five_hundred.training_env import FiveHundredVecEnv #type: ignore

_CHECKPOINT_RE = re.compile(r"_(\d+)_steps\.zip$")


def _seed_pool_from_checkpoints(pool: OpponentPool, checkpoint_dir: str, limit: int) -> None:
    """
    Preload the opponent pool with past checkpoints from disk. The pool
    only lives in process memory, so without this every resumed run starts
    out training against nothing but the current policy until snapshots
    accumulate again.

    Input:
    - pool (OpponentPool) : the pool to seed
    - checkpoint_dir (str) : directory containing ppo_<steps>_steps.zip files
    - limit (int) : max checkpoints to load, evenly spaced across the run

    Output (None)
    """
    if limit <= 0 or not os.path.isdir(checkpoint_dir):
        return

    steps_to_path = {}
    for name in os.listdir(checkpoint_dir):
        m = _CHECKPOINT_RE.search(name)
        if m:
            steps_to_path[int(m.group(1))] = os.path.join(checkpoint_dir, name)
    if not steps_to_path:
        return

    ordered = [steps_to_path[s] for s in sorted(steps_to_path)]
    stride = max(1, len(ordered) // limit)
    chosen = ordered[::stride][-limit:]

    print(f"Seeding opponent pool with {len(chosen)} past checkpoints...")
    for path in chosen:
        try:
            model = MaskablePPO.load(path, device="cpu", n_steps=1, n_envs=1)
        except Exception as exc:
            print(f"  skipping {path}: {exc}")
            continue
        pool.add(model.policy)
        del model
    print("Opponent pool seeded.")


def main() -> None:
    """
    Train a single shared policy to play 500 via self-play, using
    heuristic_bidding for bidding/discarding (see hybrid_env.py) so the
    policy only ever has to decide which card to play.

    Progress / interruption safety:
    - Checkpoints are saved every --checkpoint-freq timesteps to
      runs/<run-name>/checkpoints/, resume with --resume-from.
    - The policy is evaluated every --eval-freq timesteps; the best-scoring checkpoint is
      kept in runs/<run-name>/best_model/, separate from the periodic ones.
    - Stats available at tensorboard --logdir runs/<run-name>/tensorboard
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="ppo_five_hundred")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000,
                         help="save a checkpoint every N environment timesteps")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                         help="evaluate the policy every N environment timesteps")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--resume-from", type=str, default=None,
                         help="path to a checkpoint .zip to resume training from")
    parser.add_argument("--opponent-pool-size", type=int, default=200,
                         help="max number of frozen policy snapshots kept as self-play opponents")
    parser.add_argument("--opponent-update-freq", type=int, default=100_000,
                         help="freeze a new opponent snapshot every N environment timesteps")
    parser.add_argument("--heuristic-opponent-weight", type=float, default=0.2,
                         help="fraction of episodes played against the rule-based card player "
                              "(five_hundred/heuristic_play.py) instead of a policy snapshot")
    parser.add_argument("--seed-pool-max", type=int, default=50,
                         help="preload the opponent pool with up to N past checkpoints from "
                              "runs/<run-name>/checkpoints (0 disables)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                         help="entropy coefficient (PPO default is 0.0)")
    parser.add_argument("--target-kl", type=float, default=0.03,
                         help="stop an update epoch early once the policy has drifted this far "
                              "(PPO default is None, no limit)")
    args = parser.parse_args()

    run_dir = f"runs/{args.run_name}"
    checkpoint_dir = f"{run_dir}/checkpoints"
    tensorboard_dir = f"{run_dir}/tensorboard"

    opponent_pool = OpponentPool(
        max_size=args.opponent_pool_size, seed=args.seed, heuristic_weight=args.heuristic_opponent_weight
    )
    _seed_pool_from_checkpoints(opponent_pool, checkpoint_dir, args.seed_pool_max)
    venv = VecMonitor(FiveHundredVecEnv(num_envs=args.num_envs, opponent_pool=opponent_pool, seed=args.seed))
    eval_venv = VecMonitor(FiveHundredVecEnv(num_envs=4, opponent_pool=opponent_pool, seed=args.seed + 10_000))

    # the ~575-dim observation needs more capacity than SB3's 64x64 default
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if args.resume_from:
        model = MaskablePPO.load(
            args.resume_from, env=venv, tensorboard_log=tensorboard_dir,
            ent_coef=args.ent_coef, target_kl=args.target_kl,
        )
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            venv,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_dir,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=checkpoint_dir,
            name_prefix="ppo",
        ),
        OpponentPoolCallback(
            update_freq=max(args.opponent_update_freq // args.num_envs, 1),
        ),
        MaskableEvalCallback(
            eval_venv,
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            n_eval_episodes=args.eval_episodes,
            best_model_save_path=f"{run_dir}/best_model",
            log_path=f"{run_dir}/eval_logs",
            deterministic=True,
        ),
    ])

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume_from is None,
        tb_log_name="PPO",
    )
    model.save(f"{run_dir}/final_model")


if __name__ == "__main__":
    main()
