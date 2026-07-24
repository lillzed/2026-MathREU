import argparse
import csv
import os
from multiprocessing import Pool

from sb3_contrib import MaskablePPO  # type: ignore

from evaluate_checkpoints import _find_checkpoints, _load_for_inference
from five_hundred.constants import DEFAULT_MAX_HANDS
from five_hundred.heuristic_play import heuristic_play_action
from five_hundred.hybrid_env import play_only_env  # type: ignore

"""
Track win rate against the fixed rule-based heuristic_play.py opponent across
a run's whole checkpoint history, instead of evaluate_checkpoints.py's
checkpoint-vs-past-selves comparison.

This matters because the self-play population can converge to a shared blind
spot: if every checkpoint plays the same way, they all beat each other at
~50% regardless of whether the underlying policy actually improved, and
evaluate_checkpoints.py has no way to tell the difference. heuristic_play.py
never changes and doesn't share the RL population's blind spots, so this is
the one opponent that can confirm whether a change (e.g. reward shaping,
opponent-pool tweaks) produced a real behavioral improvement rather than just
reshuffling the self-play population.

Usage:
    python evaluate_vs_heuristic.py --run-name run5-test --interval 1000000 --games 100

Writes runs/<run-name>/eval_vs_heuristic.csv and eval_vs_heuristic.png: one
point per checkpoint, showing that checkpoint's win rate against the fixed
heuristic opponent.
"""

_worker_model_cache: dict[str, MaskablePPO] = {}


def _get_model(path: str) -> MaskablePPO:
    model = _worker_model_cache.get(path)
    if model is None:
        model = _load_for_inference(path)
        _worker_model_cache[path] = model
    return model


def _play_one_game(task: tuple[int, int, int, str]) -> int:
    """
    Play one game, a checkpoint vs. the fixed heuristic. seats are teams by
    parity (0/2 vs 1/3, see five_hundred/game.py); model_team says which team
    the checkpoint plays as this game (alternated across games to cancel out
    seat 0's fixed first-bid advantage).

    Output (int): 1 if the checkpoint's team won, 0 if the heuristic's team
    won, -1 for a draw (tied score at max_hands, rare).
    """
    seed, model_team, max_hands, model_path = task
    model = _get_model(model_path)

    env = play_only_env(max_hands=max_hands)
    env.reset(seed=seed)

    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            break
        seat = int(agent.rsplit("_", 1)[1])
        if seat % 2 == model_team:
            obs = env.observe(agent)
            action, _ = model.predict(obs["observation"], action_masks=obs["action_mask"], deterministic=True)
        else:
            action = heuristic_play_action(env.unwrapped._game)
        env.step(int(action))

    while env.agents:
        env.step(None)

    scores = env.unwrapped._game.team_scores
    if scores[model_team] == scores[1 - model_team]:
        return -1
    return 1 if scores[model_team] > scores[1 - model_team] else 0


def _evaluate_matchup(pool: Pool, model_path: str, games: int, base_seed: int, max_hands: int) -> dict:
    tasks = [(base_seed + i, i % 2, max_hands, model_path) for i in range(games)]
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


def _plot(rows: list[dict], run_name: str, output_path: str, markers: list[tuple[int, str]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    win_rates = [r["win_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, win_rates, marker="o", color="#3b82f6", linewidth=2)
    ax.axhline(50, color="#94a3b8", linestyle="--", linewidth=1, label="50% (even match)")
    for step, label in markers:
        ax.axvline(step, color="#f97316", linestyle=":", linewidth=1)
        ax.text(step, 2, f" {label}", color="#f97316", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Checkpoint (training step)")
    ax.set_ylabel("Win rate vs. heuristic_play.py (%)")
    ax.set_title(f"{run_name} vs. fixed heuristic opponent")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default="ppo_v1")
    parser.add_argument("--interval", type=int, default=1_000_000,
                         help="only evaluate checkpoints at multiples of this many steps")
    parser.add_argument("--games", type=int, default=100, help="games played per checkpoint")
    parser.add_argument("--max-hands", type=int, default=DEFAULT_MAX_HANDS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    parser.add_argument("--marker", action="append", default=[],
                         help="STEP:LABEL to annotate on the plot (e.g. a fix landing "
                              "at a given step); may be passed multiple times")
    args = parser.parse_args()

    checkpoint_dir = f"runs/{args.run_name}/checkpoints"
    steps_to_path = _find_checkpoints(checkpoint_dir)
    if not steps_to_path:
        raise SystemExit(f"no checkpoints found in {checkpoint_dir}")

    eval_steps = sorted(step for step in steps_to_path if step % args.interval == 0)
    if not eval_steps:
        raise SystemExit(f"no checkpoints at multiples of {args.interval} found in {checkpoint_dir}")

    print(f"Evaluating {len(eval_steps)} checkpoints from {args.run_name} against the fixed "
          f"heuristic opponent, {args.games} games each...")

    rows = []
    pool = Pool(processes=args.workers)
    try:
        for step in eval_steps:
            result = _evaluate_matchup(pool, steps_to_path[step], args.games, args.seed + step, args.max_hands)
            print(f"  step {step:>9}: win_rate={result['win_rate']:.3f}  "
                  f"(W{result['wins']}/L{result['losses']}/D{result['draws']})")
            rows.append({"step": step, **result})
    finally:
        pool.close()
        pool.join()

    output_csv = args.output_csv or f"runs/{args.run_name}/eval_vs_heuristic.csv"
    _write_csv(output_csv, rows)
    print(f"Wrote {output_csv}")

    markers = []
    for spec in args.marker:
        step_str, _, label = spec.partition(":")
        markers.append((int(step_str), label or step_str))

    output_plot = args.output_plot or f"runs/{args.run_name}/eval_vs_heuristic.png"
    _plot(rows, args.run_name, output_plot, markers)
    print(f"Wrote {output_plot}")


if __name__ == "__main__":
    main()
