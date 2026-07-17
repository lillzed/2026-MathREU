import argparse

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from five_hundred.training_env import FiveHundredVecEnv


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
    args = parser.parse_args()

    run_dir = f"runs/{args.run_name}"
    checkpoint_dir = f"{run_dir}/checkpoints"
    tensorboard_dir = f"{run_dir}/tensorboard"

    venv = VecMonitor(FiveHundredVecEnv(num_envs=args.num_envs, seed=args.seed))
    eval_venv = VecMonitor(FiveHundredVecEnv(num_envs=4, seed=args.seed + 10_000))

    if args.resume_from:
        model = MaskablePPO.load(args.resume_from, env=venv, tensorboard_log=tensorboard_dir)
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            venv,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_dir,
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=checkpoint_dir,
            name_prefix="ppo",
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
