# Reinforcement Learning for the Card Game 500

A Math REU project on applying reinforcement learning to **500** (a
trick-taking, imperfect-information card game popular in Australia and
related to Euchre). The project builds a full game engine and RL
environment from scratch, then trains a self-play agent with maskable PPO
to bid, discard, and play against heuristic and Monte Carlo opponents.

It progresses in three stages, each in its own directory:

1. **[`rl-fundamentals/`](rl-fundamentals)** — from-scratch implementations of
   classic RL algorithms (bandits, dynamic programming, Monte Carlo and
   TD learning) used to build intuition before tackling a full card game.
2. **[`data_collection/`](data_collection)** — a vendored, headless build of a
   [command-line 500 implementation](data_collection/500) is used to generate
   bot-vs-bot games, which are parsed into structured, per-decision datasets.
3. **[`five_hundred/`](five_hundred)** — a Python game
   engine and [PettingZoo](https://pettingzoo.farama.org/) environment for 500,
   with a heuristic bidding/play baseline, a perfect-information Monte Carlo
   (PIMC) bot, and self-play PPO training (`train.py`) built on
   [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) /
   [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib).

## Repository layout

```
.
├── five_hundred/            # game engine, RL environment, heuristics, training env
│   ├── game.py              #   core rules/state machine (bidding, kitty, trick play, scoring)
│   ├── cards.py, encoding.py#   card representation and observation/action encoding
│   ├── env.py                #   PettingZoo AECEnv wrapping FiveHundredGame
│   ├── hybrid_env.py          #   wraps env.py so bidding/discarding are handled by
│   │                          #     heuristic_bidding.py, leaving only card play to the policy
│   ├── heuristic_bidding.py, heuristic_play.py  # rule-based bidding/play baseline & opponent
│   ├── mc_bot.py             #   perfect-information Monte Carlo (PIMC) card-play bot
│   ├── opponent_pool.py, callbacks.py  # self-play opponent pool of frozen policy snapshots
│   └── training_env.py       #   vectorized SB3 VecEnv driving many self-play games in parallel
├── train.py                  # trains a MaskablePPO policy via self-play
├── evaluate_checkpoints.py   # win rate of a checkpoint vs. its own past checkpoints over training
├── evaluate_vs_heuristic.py  # win rate of every checkpoint in a run vs. the fixed heuristic bot
├── play_vs_bots.py           # play an interactive game against trained/heuristic bots
├── data_collection/          # earlier data-generation phase, see its own README
│   ├── 500/                  #   vendored command-line 500 engine (bots + server/client)
│   ├── collect_and_parse.py  #   runs bot games and parses logs into JSONL decision records
│   ├── rules.py               #   pure-Python port of the C engine's legality rules
│   └── data/                  #   collected datasets (games.jsonl, decisions.jsonl)
├── rl-fundamentals/           # warm-up RL implementations, see its own README
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Python 3.11+ is recommended (the code uses `X | None` style type hints).

## Training the agent

```bash
python train.py --timesteps 5_000_000 --run-name my_run
```

This trains one shared `MaskablePPO` policy to play the card-play phase of
500. Bidding and discarding are always handled by the rule-based
`heuristic_bidding` module (see `five_hundred/hybrid_env.py`), so the policy's
only job is choosing which card to play. The non-learner seats are driven by
an `OpponentPool` (`five_hundred/opponent_pool.py`) that, by default, is the
PIMC bot (`five_hundred/mc_bot.py`); it can optionally mix in frozen snapshots
of the training policy itself for true self-play. Checkpoints, the
best-scoring model, and TensorBoard logs are written under
`runs/<run-name>/` (`tensorboard --logdir runs/<run-name>/tensorboard`).
Training can be resumed from any checkpoint with `--resume-from`.

See `python train.py --help` for the full set of options (opponent pool size,
anchor weight, entropy coefficient, etc.), and the module docstring in
`train.py` for more detail.

## Evaluating a run

```bash
python evaluate_checkpoints.py --run-name my_run --interval 500000 --games 200
python evaluate_vs_heuristic.py --run-name my_run --interval 500000 --games 200
```

Self-play reward nets to roughly zero and isn't a useful progress signal by
itself, so these scripts instead measure a checkpoint's win rate against
either its own past checkpoints or the fixed heuristic bot, and plot the
result over the course of training.

## Playing against the trained agent

```bash
python play_vs_bots.py --run-name my_run
```

Starts an interactive command-line game: you play one seat, the other three
are bot-controlled (heuristic bidding/discarding, and card play from either
the loaded checkpoint or a heuristic, depending on setup).

## Technical notes

- **Environment**: `five_hundred/env.py` implements a
  [PettingZoo](https://pettingzoo.farama.org/) `AECEnv` for 500's full state
  machine (bidding → kitty exchange → 10 tricks of card play → scoring →
  next hand), with a 575-dimension observation and a 71-action discrete
  action space (43 cards + pass + 25 bid values + misère + open misère; see
  `five_hundred/encoding.py`) plus a per-step action mask for illegal moves.
- **`hybrid_env.py`** narrows that general environment to just the card-play
  decision, resolving bidding/discarding with `heuristic_bidding.py` — this
  is what `train.py` actually trains against.
- **Self-play**: `five_hundred/opponent_pool.py` keeps a pool of frozen
  policy snapshots (periodically pushed by `OpponentPoolCallback`) plus a
  fixed anchor opponent, so training against a moving target stays stable.
- **`five_hundred/mc_bot.py`** is a determinization-based Monte Carlo bot:
  it samples several full deals consistent with public information (own
  hand, cards played, inferred suit voids), plays each one out with the
  heuristic policy, and picks the card that scored best on average.

## Data collection phase

Before the RL environment existed, `data_collection/` was used to generate a
labeled dataset of real bot decisions (bid/discard/play, with hand contents
and legal actions reconstructed from the game's text log) by running many
games of the vendored command-line implementation and parsing its output.
See [`data_collection/README.md`](data_collection/README.md) for details —
this dataset informed the design of the heuristic bidding/play baselines
used as the training opponent above.

## RL fundamentals

`rl-fundamentals/` contains earlier, self-contained explorations of
foundational RL methods (multi-armed bandits, dynamic programming on
FrozenLake, and Monte Carlo/TD methods on a custom cliff-walking maze) done
before building the full 500 project. See
[`rl-fundamentals/README.md`](rl-fundamentals/README.md).
