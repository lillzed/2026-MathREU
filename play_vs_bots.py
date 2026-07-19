import argparse
import glob
import os
import re
import sys

from sb3_contrib import MaskablePPO  # type: ignore

from five_hundred import cards
from five_hundred import render as trace
from five_hundred.constants import DEFAULT_MAX_HANDS, NUM_PLAYERS, Phase
from five_hundred.env import env as base_env
from five_hundred.hybrid_env import _heuristic_action

"""
Play a full game of 500 against 3 bot-controlled seats. The bots bid and
discard with heuristic_bidding and play cards with the loaded MaskablePPO
checkpoint -- the same setup they were trained under.

Usage:
    python play_vs_bots.py                                  # most recently trained checkpoint, you're Player 0
    python play_vs_bots.py --run-name run5-test              # latest checkpoint from a specific run
    python play_vs_bots.py --run-name run5-test --checkpoint 8000000
    python play_vs_bots.py --model runs/run5-test/best_model.zip --seat 2
    python play_vs_bots.py --matches 5 --seed 0               # play several matches in a row

Type a menu number at any prompt, or 'q' to quit.
"""

_CHECKPOINT_RE = re.compile(r"_(\d+)_steps\.zip$")


def _latest_checkpoint_anywhere() -> str:
    candidates = glob.glob("runs/*/checkpoints/*.zip")
    if not candidates:
        raise SystemExit("no checkpoints found under runs/ -- pass --model explicitly")
    return max(candidates, key=os.path.getmtime)


def _resolve_model_path(args: argparse.Namespace) -> str:
    if args.model:
        return args.model

    if args.run_name:
        checkpoint_dir = f"runs/{args.run_name}/checkpoints"
        steps_to_path = {}
        for name in os.listdir(checkpoint_dir):
            m = _CHECKPOINT_RE.search(name)
            if m:
                steps_to_path[int(m.group(1))] = os.path.join(checkpoint_dir, name)
        if not steps_to_path:
            raise SystemExit(f"no checkpoints found in {checkpoint_dir}")
        if args.checkpoint is not None:
            if args.checkpoint not in steps_to_path:
                raise SystemExit(f"no checkpoint at step {args.checkpoint} in {checkpoint_dir}")
            return steps_to_path[args.checkpoint]
        return steps_to_path[max(steps_to_path)]

    return _latest_checkpoint_anywhere()


def _format_hand(hand) -> str:
    ordered = sorted(hand, key=lambda c: (cards.suit_of(c), -cards.rank_of(c)))
    return "  ".join(cards.format_card(c) for c in ordered)


def _prompt_choice(header: str, labels: list[str], per_row: int = 6) -> int:
    print(header)
    for row_start in range(0, len(labels), per_row):
        row = labels[row_start:row_start + per_row]
        print("  " + "   ".join(f"{i}) {label}" for i, label in enumerate(row, start=row_start)))
    while True:
        raw = input("> ").strip().lower()
        if raw in ("q", "quit"):
            raise SystemExit("Quitting.")
        if raw.isdigit() and 0 <= int(raw) < len(labels):
            return int(raw)
        print(f"Enter a number from 0 to {len(labels) - 1} (or 'q' to quit).")


def _human_bid(game) -> int:
    if game.bet_winner is None:
        print("No bids yet.")
    else:
        current = trace._contract_description({
            "misere": game.misere, "open_misere": game.open_misere,
            "highest_bet": game.highest_bet, "highest_suit": game.highest_suit,
        })
        print(f"Current highest bid: {current} (Player {game.bet_winner})")
    print(f"Your hand: {_format_hand(game.hands[game.current_player])}")
    legal = game.legal_actions()
    labels = [trace._bid_description(a) for a in legal]
    return legal[_prompt_choice("Choose your bid:", labels)]


def _human_discard(game) -> int:
    print(f"Your hand: {_format_hand(game.hands[game.current_player])}")
    legal = sorted(game.legal_actions(), key=lambda c: (cards.suit_of(c), -cards.rank_of(c)))
    labels = [cards.format_card(c) for c in legal]
    return legal[_prompt_choice("Choose a card to discard (3 total):", labels)]


def _human_play(game) -> int:
    trump_name = "no trump" if game.trump == cards.NO_SUIT else trace._SUIT_CHARS[game.trump]
    print(f"\nTrick {game.tricks_played + 1}/10  (trump: {trump_name})")
    if game.current_trick:
        for player, card in game.current_trick:
            print(f"  Player {player} played {cards.format_card(card)}")
    else:
        print("  You lead this trick.")
    print(f"Your hand: {_format_hand(game.hands[game.current_player])}")
    legal = sorted(game.legal_actions(), key=lambda c: (cards.suit_of(c), -cards.rank_of(c)))
    labels = [cards.format_card(c) for c in legal]
    return legal[_prompt_choice("Choose a card to play:", labels)]


def _human_action(game) -> int:
    if game.phase == Phase.BIDDING:
        return _human_bid(game)
    if game.phase == Phase.DISCARD:
        return _human_discard(game)
    return _human_play(game)


def _play_match(model: MaskablePPO, human_seat: int, seed: int | None, max_hands: int) -> None:
    e = base_env(render_mode="human", max_hands=max_hands)
    e.reset(seed=seed)
    human_agent = f"player_{human_seat}"
    partner_seat = (human_seat + 2) % NUM_PLAYERS
    opponents = [s for s in range(NUM_PLAYERS) if s not in (human_seat, partner_seat)]
    print(f"\nYou are Player {human_seat}, partnered with Player {partner_seat} (bot). "
          f"Players {opponents} are the opposing bots.")

    while e.agents:
        agent = e.agent_selection
        if e.terminations[agent] or e.truncations[agent]:
            e.step(None)
            continue

        game = e.unwrapped._game
        if agent == human_agent:
            action = _human_action(game)
        elif game.phase in (Phase.BIDDING, Phase.DISCARD):
            action = _heuristic_action(game)
        else:
            obs = e.observe(agent)
            action, _ = model.predict(obs["observation"], action_masks=obs["action_mask"], deterministic=True)
            action = int(action)
        e.step(action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str, default=None,
                         help="path to a checkpoint .zip; overrides --run-name/--checkpoint")
    parser.add_argument("--run-name", type=str, default=None,
                         help="load from runs/<run-name>/checkpoints; defaults to the most "
                              "recently modified checkpoint across every run")
    parser.add_argument("--checkpoint", type=int, default=None,
                         help="specific step to load from --run-name (defaults to its highest step)")
    parser.add_argument("--seat", type=int, default=0, choices=range(NUM_PLAYERS),
                         help="your seat (0-3); seats 0/2 and 1/3 are partnered teams")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--matches", type=int, default=1, help="how many full matches (to 500) to play in a row")
    parser.add_argument("--max-hands", type=int, default=DEFAULT_MAX_HANDS)
    args = parser.parse_args()

    model_path = _resolve_model_path(args)
    print(f"Loading {model_path} ...")
    model = MaskablePPO.load(model_path, device="cpu", n_steps=1, n_envs=1)

    for i in range(args.matches):
        if args.matches > 1:
            print(f"\n########## Match {i + 1}/{args.matches} ##########")
        seed = None if args.seed is None else args.seed + i
        _play_match(model, args.seat, seed, args.max_hands)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
