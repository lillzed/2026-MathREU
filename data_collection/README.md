# Data Collection

Before the `five_hundred/` RL environment existed, this directory was used to
build a labeled dataset of real 500 decisions by running many bot-vs-bot
games through a command-line implementation of the game and parsing its text
output.

## Contents

- **`500/`** (not included — see Setup below) —
  [Gareth001/500](https://github.com/Gareth001/500), a command-line/C
  implementation of 500 with built-in bots. Used here purely as a headless,
  offline game engine — not for networked play — to generate legal,
  rules-correct games between bots.
- **`rules.py`** — a pure-Python port of just the rule logic needed to
  reconstruct legal actions from the C engine's text log (bid legality,
  follow-suit legality), mirroring `server.c`/`cards.c` so decision-level
  data can be rebuilt without depending on the C server at runtime.
- **`collect_and_parse.py`** — runs N games against the compiled `500`
  server/bots, captures their console output, and parses it into two
  related JSONL tables:
  - `data/games.jsonl` — one row per completed game (final scores, winner,
    number of hands played)
  - `data/decisions.jsonl` — one row per individual player decision (bid,
    discard, or card play), including that player's hand, the legal actions
    available, and the action actually taken
- **`data/`** — the collected datasets described above.

## Setup

This directory expects a compiled copy of
[Gareth001/500](https://github.com/Gareth001/500) at `data_collection/500/`.
It isn't vendored in this repo (that project has no license, so its source
isn't redistributed here) — clone and build it yourself:

```bash
git clone https://github.com/Gareth001/500 500
cd 500 && make && cd ..
```

Requires `gcc`/`make` (e.g. via MinGW on Windows). `data_collection/500/` is
gitignored, so nothing under it gets committed.

Then collect and parse games:

```bash
python collect_and_parse.py --games 500
python collect_and_parse.py --games 5 --keep-raw --review 3   # keep raw logs + pretty-print a few hands for QA
```

Run from inside `data_collection/` — `collect_and_parse.py`'s default paths
(`./500/server`, `data/games.jsonl`, `data/decisions.jsonl`) are relative to
this directory. This dataset informed the design of the heuristic
bidding/play baselines (`five_hundred/heuristic_bidding.py`,
`five_hundred/heuristic_play.py`) used in the main project.
