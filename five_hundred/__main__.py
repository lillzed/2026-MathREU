"""Smoke test: play one full match with uniformly-random legal actions on
every seat and report the outcome.

Run with `python -m five_hundred` for a quiet pass/fail check, or
`python -m five_hundred --render` to print a human-readable play-by-play
trace of every bid/discard/joker-suit/play decision - useful for eyeballing
that bidding, follow-suit, trump, and scoring actually match the real rules.

This only exercises the engine/env plumbing (legality, turn order, scoring,
termination) - it says nothing about play quality, since the actions are
random.
"""

import argparse
import random

from .env import env
from .constants import WIN_POINTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="print a play-by-play trace")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    e = env(render_mode="human" if args.render else None)
    e.reset(seed=args.seed)

    steps = 0

    for agent in e.agent_iter():
        observation, reward, termination, truncation, info = e.last()
        steps += 1

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            legal = [i for i, ok in enumerate(mask) if ok]
            action = random.choice(legal)

        e.step(action)

    game = e.unwrapped._game
    hands_played = game.hand_number
    scores = game.team_scores

    print(f"\nsteps={steps} hands_played={hands_played} final_scores={scores} "
          f"match_over={not game.truncated} truncated={game.truncated}")

    # a hand's points aren't capped at the threshold, so the final score can
    # overshoot past +-WIN_POINTS - the invariant is just that it crossed it
    assert game.done
    assert game.truncated or any(abs(s) >= WIN_POINTS for s in scores), scores


if __name__ == "__main__":
    main()
