import argparse
import csv
import os
import re
from multiprocessing import Pool

from sb3_contrib import MaskablePPO  # type: ignore

from five_hundred.constants import DEFAULT_MAX_HANDS
from five_hundred.hybrid_env import play_only_env  # type: ignore

"""
Evaluate a checkpoint's win rate against a set of its own past checkpoints,
to track skill progress independent of the co-training opponent (see
train.py: reward in self-play nets to ~0 and isn't itself a progress signal).

Usage:
    python evaluate_checkpoints.py --run-name ppo_v1 --interval 500000 --games 200

Writes runs/<run-name>/eval_vs_past.csv and eval_vs_past.png: one point per
past checkpoint, showing the current model's win rate against it.
"""

_CHECKPOINT_RE = re.compile(r"_(\d+)_steps\.zip$")

_worker_current: MaskablePPO
_worker_past_cache: dict[str, MaskablePPO] = {}


def _load_for_inference(path: str) -> MaskablePPO:
    """
    Load a checkpoint for predict()-only use. Plain MaskablePPO.load()
    rebuilds a full rollout buffer sized for the training config baked into
    the checkpoint (n_steps x n_envs, e.g. 2048 x 16) even though learn()
    is never called here -- with many worker processes each caching several
    checkpoints, that can exhaust memory. Overriding n_steps/n_envs shrinks
    the buffer to negligible size.
    """
    return MaskablePPO.load(path, device="cpu", n_steps=1, n_envs=1)


def _init_worker(current_path: str) -> None:
    global _worker_current
    _worker_current = _load_for_inference(current_path)


def _play_one_game(task: tuple[int, int, int, str]) -> int:
    """
    Play one game, current model vs. a past checkpoint. seats are teams by
    parity (0/2 vs 1/3, see five_hundred/game.py); current_team says which
    team the current model plays as this game (alternated across games to
    cancel out seat 0's fixed first-bid advantage, since the game always
    starts with start_player=0).

    Output (int): 1 if the current model's team won, 0 if the past
    checkpoint's team won, -1 for a draw (tied score at max_hands, rare).
    """
    seed, current_team, max_hands, past_path = task

    past_model = _worker_past_cache.get(past_path)
    if past_model is None:
        past_model = _load_for_inference(past_path)
        _worker_past_cache[past_path] = past_model

    model_for_team = {current_team: _worker_current, 1 - current_team: past_model}

    env = play_only_env(max_hands=max_hands)
    env.reset(seed=seed)

    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            break
        obs = env.observe(agent)
        seat = int(agent.rsplit("_", 1)[1])
        model = model_for_team[seat % 2]
        action, _ = model.predict(obs["observation"], action_masks=obs["action_mask"], deterministic=True)
        env.step(int(action))

    while env.agents:
        env.step(None)

    scores = env.unwrapped._game.team_scores
    if scores[current_team] == scores[1 - current_team]:
        return -1
    return 1 if scores[current_team] > scores[1 - current_team] else 0


def _find_checkpoints(checkpoint_dir: str) -> dict[int, str]:
    steps_to_path = {}
    for name in os.listdir(checkpoint_dir):
        m = _CHECKPOINT_RE.search(name)
        if m:
            steps_to_path[int(m.group(1))] = os.path.join(checkpoint_dir, name)
    return steps_to_path


def _evaluate_matchup(pool: Pool, past_path: str, games: int, base_seed: int, max_hands: int) -> dict:
    tasks = [(base_seed + i, i % 2, max_hands, past_path) for i in range(games)]
    results = pool.map(_play_one_game, tasks)

    wins = results.count(1)
    losses = results.count(0)
    draws = results.count(-1)
    win_rate = (wins + 0.5 * draws) / games
    return {"win_rate": win_rate, "wins": wins, "losses": losses, "draws": draws}


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "win_rate", "wins", "losses", "draws"])
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], current_label: str, output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    win_rates = [r["win_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, win_rates, marker="o", color="#3b82f6", linewidth=2)
    ax.axhline(50, color="#94a3b8", linestyle="--", linewidth=1, label="50% (even match)")
    ax.set_xlabel("Opponent checkpoint (training step)")
    ax.set_ylabel(f"{current_label} win rate (%)")
    ax.set_title(f"{current_label} vs. past checkpoints")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default="ppo_v1")
    parser.add_argument("--current", type=str, default=None,
                         help="checkpoint being evaluated; defaults to the highest-step "
                              "checkpoint in runs/<run-name>/checkpoints")
    parser.add_argument("--interval", type=int, default=500_000,
                         help="only evaluate against past checkpoints at multiples of this many steps")
    parser.add_argument("--games", type=int, default=200, help="games played per matchup")
    parser.add_argument("--max-hands", type=int, default=DEFAULT_MAX_HANDS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    args = parser.parse_args()

    checkpoint_dir = f"runs/{args.run_name}/checkpoints"
    steps_to_path = _find_checkpoints(checkpoint_dir)
    if not steps_to_path:
        raise SystemExit(f"no checkpoints found in {checkpoint_dir}")

    current_path = args.current or steps_to_path[max(steps_to_path)]
    path_to_step = {path: step for step, path in steps_to_path.items()}
    current_step = path_to_step.get(current_path)
    current_label = f"step {current_step}" if current_step is not None else os.path.basename(current_path)

    past_steps = sorted(
        step for step, path in steps_to_path.items()
        if step % args.interval == 0 and path != current_path
    )
    if not past_steps:
        raise SystemExit(f"no past checkpoints at multiples of {args.interval} found in {checkpoint_dir}")

    print(f"Evaluating {current_path} ({current_label}) against {len(past_steps)} "
          f"past checkpoints, {args.games} games each...")

    rows = []
    pool = Pool(processes=args.workers, initializer=_init_worker, initargs=(current_path,))
    try:
        for step in past_steps:
            past_path = steps_to_path[step]
            result = _evaluate_matchup(pool, past_path, args.games, args.seed + step, args.max_hands)
            print(f"  vs step {step:>9}: win_rate={result['win_rate']:.3f}  "
                  f"(W{result['wins']}/L{result['losses']}/D{result['draws']})")
            rows.append({"step": step, **result})
    finally:
        pool.close()
        pool.join()

    output_csv = args.output_csv or f"runs/{args.run_name}/eval_vs_past.csv"
    _write_csv(output_csv, rows)
    print(f"Wrote {output_csv}")

    output_plot = args.output_plot or f"runs/{args.run_name}/eval_vs_past.png"
    _plot(rows, current_label, output_plot)
    print(f"Wrote {output_plot}")


if __name__ == "__main__":
    main()
