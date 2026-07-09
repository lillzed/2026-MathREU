import argparse
import random

from .env import env
from .constants import WIN_POINTS


def main() -> None:
    """
    Play one match of the 500 RL env end to end, choosing a uniformly random legal
    action for every agent each step. Serves as a smoke test / usage example for the
    AECEnv API.

    Input (None)

    Output (None)
    """
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

    assert game.done
    assert game.truncated or any(abs(s) >= WIN_POINTS for s in scores), scores


if __name__ == "__main__":
    main()