"""
500 Card Game - Data Collection & Parser
=========================================
Runs N bot games, captures output, and parses into two related JSONL
tables suitable for RL training:

    data/games.jsonl     - one row per completed game
    data/decisions.jsonl - one row per individual player decision
                            (bid / discard / play)

Hand state (each player's own hand at decision time) and legal actions
are reconstructed in Python from the server's text log - see rules.py.
Hands that never reach the play phase (everyone passes, hand is
redealt) are skipped: no player's hand is ever revealed in the log for
those, so there is nothing to reconstruct.

Usage:
    python collect_and_parse.py --games 500
    python collect_and_parse.py --games 5 --keep-raw --review 3
"""

import subprocess
import os
import re
import json
import argparse
from datetime import datetime, timezone

import rules


# ─────────────────────────────────────────
# STEP 1: Run games and capture output
# ─────────────────────────────────────────

def run_games(n, server_path, port, password, bot_types, raw_dir=None):
    server_path = os.path.normpath(server_path)  # CreateProcess on Windows
    # doesn't resolve relative paths containing forward slashes
    results = []
    for i in range(n):
        print(f"Running game {i + 1}/{n}...", end="\r")
        timestamp = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(
            [server_path, str(port), password, bot_types],
            capture_output=True,
            text=True
        )
        output = result.stdout
        results.append((output, timestamp))

        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)
            with open(os.path.join(raw_dir, f"game_{i+1}.txt"), "w") as f:
                f.write(output)

    print(f"\nDone! Ran {n} games.")
    return results


# ─────────────────────────────────────────
# STEP 2: Parse raw output into structured data
# ─────────────────────────────────────────

def parse_output(raw_text):
    """Parse one server run (one or more hands) into a list of hand dicts."""
    hands = []
    lines = raw_text.strip().split("\n")

    current_hand = None
    current_trick = None
    hand_number = 0
    trick_number = 0
    bid_order = 0

    for line in lines:
        line = line.strip()

        if line == "Shuffling and Dealing":
            if current_hand is not None:
                hands.append(current_hand)
            hand_number += 1
            trick_number = 0
            bid_order = 0
            current_hand = {
                "hand_number": hand_number,
                "bids": [],
                "winning_bid": None,
                "kitty": None,
                "discards": [],
                "joker_suit": None,
                "tricks": [],
                "outcome": None,
                "score_team0": None,
                "score_team1": None,
            }
            current_trick = None

        elif (m := re.match(r"Player (\d+) passed", line)):
            if current_hand:
                bid_order += 1
                current_hand["bids"].append({
                    "order": bid_order, "player": int(m.group(1)),
                    "action": "pass", "bid_value": None
                })

        elif (m := re.match(r"Player (\d+) bet (\S+)", line)):
            if current_hand:
                bid_order += 1
                current_hand["bids"].append({
                    "order": bid_order, "player": int(m.group(1)),
                    "action": "bet", "bid_value": m.group(2)
                })

        elif (m := re.match(r"Player (\d+) won the bet with (\S+)!", line)):
            if current_hand:
                current_hand["winning_bid"] = {"player": int(m.group(1)), "bid": m.group(2)}

        elif (m := re.match(r"Kitty: (\S+) (\S+) (\S+)", line)):
            if current_hand:
                current_hand["kitty"] = [m.group(1), m.group(2), m.group(3)]

        elif (m := re.match(r"Discarded: (\S+)", line)):
            if current_hand:
                current_hand["discards"].append(m.group(1))

        elif (m := re.match(r"Joker suit chosen\. It is a (\S)", line)):
            if current_hand:
                current_hand["joker_suit"] = m.group(1)

        elif (m := re.match(r"Player (\d+) played (\S+)\. Player (\d+) (winning|won) with (\S+)", line)):
            if current_hand is None:
                continue

            player = int(m.group(1))
            card = m.group(2)
            status = m.group(4)
            trick_winner = int(m.group(3))
            winning_card = m.group(5)

            if current_trick is None:
                trick_number += 1
                current_trick = {
                    "trick_number": trick_number,
                    "plays": [],
                    "winner": None,
                    "winning_card": None,
                    "betting_team_tricks": None,
                }

            current_trick["plays"].append({"player": player, "card": card})

            if status == "won":
                current_trick["winner"] = trick_winner
                current_trick["winning_card"] = winning_card
                current_hand["tricks"].append(current_trick)
                current_trick = None

        elif (m := re.match(r"Betting team has won (\d+) trick\(s\)", line)):
            if current_hand and current_hand["tricks"]:
                current_hand["tricks"][-1]["betting_team_tricks"] = int(m.group(1))

        elif line == "Betting team won!":
            if current_hand:
                current_hand["outcome"] = "won"

        elif line == "Betting team lost!":
            if current_hand:
                current_hand["outcome"] = "lost"

        elif (m := re.match(r"Team 0 points: (-?\d+)", line)):
            if current_hand:
                current_hand["score_team0"] = int(m.group(1))

        elif (m := re.match(r"Team 1 points: (-?\d+)", line)):
            if current_hand:
                current_hand["score_team1"] = int(m.group(1))

    if current_hand is not None:
        hands.append(current_hand)

    return hands


# ─────────────────────────────────────────
# STEP 3: Reconstruct per-player hands
# ─────────────────────────────────────────

NUM_PLAYERS = 4
WIN_POINTS = 500


def reconstruct_hand(hand):
    """
    Recovers each player's 10-card hand at the start of bidding, purely
    from cards the log already reveals (see plan / module docstring).
    Returns None for hands that never reached a winning bid (redealt) or
    a misere/open-misere contract (a player sits out play entirely, which
    breaks the "hand = cards eventually played" reconstruction - bots
    never bid misere, so this is not expected to occur in practice).
    """
    winning_bid = hand.get("winning_bid")
    if not winning_bid:
        return None

    final_bid = rules.normalize_bid_token(winning_bid["bid"])
    if final_bid in ("MISERE", "OPENMISERE"):
        return None

    bet_winner = winning_bid["player"]

    played_by_player = {p: [] for p in range(NUM_PLAYERS)}
    for trick in hand["tricks"]:
        for play in trick["plays"]:
            played_by_player[play["player"]].append(play["card"])

    kitty = hand.get("kitty") or []
    discards = [c for c in hand.get("discards") or [] if c]

    if len(kitty) != 3 or len(discards) != 3:
        return None

    # bot.c's get_lowest_non_trump_card can log a non-existent "0S"
    # sentinel discard for trump-heavy hands (see rules.is_valid_card) -
    # unreconstructable, skip the hand rather than guessing.
    if not all(rules.is_valid_card(c) for c in kitty + discards):
        return None

    initial_hands = {}
    for p in range(NUM_PLAYERS):
        if p == bet_winner:
            played = set(played_by_player[p])
            initial_hands[p] = (played | set(discards)) - set(kitty)
        else:
            initial_hands[p] = set(played_by_player[p])

    # sanity check: every player should have started with exactly 10
    # cards, and a full deck (10*4 + kitty) accounts for all 43 cards.
    if any(len(h) != 10 for h in initial_hands.values()):
        return None
    all_cards = set().union(*initial_hands.values()) | set(kitty)
    if len(all_cards) != 43:
        return None

    return {
        "initial_hands": initial_hands,
        "bet_winner": bet_winner,
        "kitty": kitty,
        "discards": discards,
        "final_bid": final_bid,
    }


def trump_suit_for_bid(final_bid):
    if final_bid in ("MISERE", "OPENMISERE"):
        return 'N'
    return final_bid[-1]


# ─────────────────────────────────────────
# STEP 4: Build decision rows (the actual RL training data)
# ─────────────────────────────────────────

def build_decisions(game_id, hand_id, hand, recon, prev_team_points):
    decisions = []
    decision_id = 0

    bid_history = []
    highest_bet, highest_suit = 0, 'S'
    misere, open_ = False, False

    bet_winner = recon["bet_winner"]
    final_bid = recon["final_bid"]
    trump = trump_suit_for_bid(final_bid)
    joker_suit = hand.get("joker_suit")

    current_hands = {p: set(recon["initial_hands"][p]) for p in range(NUM_PLAYERS)}

    # ---- bid phase ----
    for bid in hand["bids"]:
        player = bid["player"]
        action_taken = "PASS" if bid["action"] == "pass" else rules.normalize_bid_token(bid["bid_value"])
        legal = rules.legal_bid_actions(highest_bet, highest_suit, misere, open_)

        current_bid = None
        if highest_bet:
            current_bid = "OPENMISERE" if open_ else "MISERE" if misere else f"{highest_bet}{highest_suit}"

        decisions.append({
            "game_id": game_id, "hand_id": hand_id, "trick_num": 0,
            "decision_id": decision_id, "player_id": player, "phase": "bid",
            "hand_cards": sorted(current_hands[player], key=rules.card_sort_key),
            "trump_suit": None, "current_bid": current_bid,
            "bid_history": [list(b) for b in bid_history],
            "cards_played_this_trick": [], "cards_played_history": [],
            "legal_actions": legal, "action_taken": action_taken,
            "action_logprob": None, "action_value": None,
            "reward": 0, "done": False,
        })
        decision_id += 1
        bid_history.append([player, action_taken])

        if action_taken == "MISERE":
            highest_bet, highest_suit, misere = 7, 'N', True
        elif action_taken == "OPENMISERE":
            highest_bet, highest_suit, misere, open_ = 10, 'D', True, True
        elif action_taken != "PASS":
            highest_bet, highest_suit = int(action_taken[:-1]), action_taken[-1]

    # ---- discard phase ----
    current_hands[bet_winner] |= set(recon["kitty"])
    for card in recon["discards"]:
        legal = rules.legal_discard_actions(sorted(current_hands[bet_winner], key=rules.card_sort_key))
        decisions.append({
            "game_id": game_id, "hand_id": hand_id, "trick_num": 0,
            "decision_id": decision_id, "player_id": bet_winner, "phase": "discard",
            "hand_cards": sorted(current_hands[bet_winner], key=rules.card_sort_key),
            "trump_suit": trump, "current_bid": final_bid,
            "bid_history": [list(b) for b in bid_history],
            "cards_played_this_trick": [], "cards_played_history": [],
            "legal_actions": legal, "action_taken": card,
            "action_logprob": None, "action_value": None,
            "reward": 0, "done": False,
        })
        decision_id += 1
        current_hands[bet_winner].discard(card)

    # ---- play phase ----
    cards_played_history = []
    for trick in hand["tricks"]:
        trick_num = trick["trick_number"]
        cards_played_this_trick = []
        lead = None

        for play in trick["plays"]:
            player, card = play["player"], play["card"]
            legal = rules.legal_play_actions(
                sorted(current_hands[player], key=rules.card_sort_key), trump, lead, joker_suit)

            decisions.append({
                "game_id": game_id, "hand_id": hand_id, "trick_num": trick_num,
                "decision_id": decision_id, "player_id": player, "phase": "play",
                "hand_cards": sorted(current_hands[player], key=rules.card_sort_key),
                "trump_suit": trump, "current_bid": final_bid,
                "bid_history": [list(b) for b in bid_history],
                "cards_played_this_trick": list(cards_played_this_trick),
                "cards_played_history": list(cards_played_history),
                "legal_actions": legal, "action_taken": card,
                "action_logprob": None, "action_value": None,
                "reward": 0, "done": False,
            })
            decision_id += 1
            current_hands[player].discard(card)

            if lead is None:
                lead = rules.effective_suit(card, trump, joker_suit)
            cards_played_this_trick.append([player, card])
            cards_played_history.append([trick_num, player, card])

    # ---- reward / done: per-hand episode, credited on each player's own
    # last decision in the hand (see plan: team point deltas aren't
    # symmetric, so only one globally-last row would lose a signal) ----
    last_decision_idx = {}
    for i, row in enumerate(decisions):
        last_decision_idx[row["player_id"]] = i

    deltas = {
        0: hand["score_team0"] - prev_team_points[0],
        1: hand["score_team1"] - prev_team_points[1],
    }
    for player, idx in last_decision_idx.items():
        decisions[idx]["done"] = True
        decisions[idx]["reward"] = deltas[player % 2]

    return decisions


# ─────────────────────────────────────────
# STEP 5: Build the games table
# ─────────────────────────────────────────

def build_games_row(game_id, timestamp, hands):
    completed = [h for h in hands if h.get("winning_bid") and h.get("score_team0") is not None]

    team_points = {0: 0, 1: 0}
    if completed:
        last = completed[-1]
        team_points = {0: last["score_team0"], 1: last["score_team1"]}

    winner = None
    if team_points[0] >= WIN_POINTS or team_points[1] <= -WIN_POINTS:
        winner = "team_0"
    elif team_points[1] >= WIN_POINTS or team_points[0] <= -WIN_POINTS:
        winner = "team_1"

    return {
        "game_id": game_id,
        "num_players": NUM_PLAYERS,
        "timestamp": timestamp,
        "final_team_scores": {"team_0": team_points[0], "team_1": team_points[1]},
        "winner": winner,
        "num_hands_played": len(completed),
    }


def process_game(game_id, raw_text, timestamp):
    hands = parse_output(raw_text)

    games_row = build_games_row(game_id, timestamp, hands)

    decisions = []
    prev_team_points = {0: 0, 1: 0}
    hand_id = 0
    skipped = 0
    for hand in hands:
        if not (hand.get("winning_bid") and hand.get("score_team0") is not None):
            continue  # redeal, or hand never finished (shouldn't happen mid-stream)

        hand_id += 1
        recon = reconstruct_hand(hand)
        if recon is None:
            skipped += 1
        else:
            decisions.extend(build_decisions(game_id, hand_id, hand, recon, prev_team_points))

        prev_team_points = {0: hand["score_team0"], 1: hand["score_team1"]}

    return games_row, decisions, skipped


# ─────────────────────────────────────────
# STEP 6: Write JSONL
# ─────────────────────────────────────────

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ─────────────────────────────────────────
# Manual review helper
# ─────────────────────────────────────────

def print_review(decisions, n_hands):
    """Rows for a given hand are contiguous (built and appended one hand at
    a time), so once n_hands distinct hands have been started it's safe to
    stop entirely on the first row of the next one."""
    shown_keys = []
    for row in decisions:
        key = (row["game_id"], row["hand_id"])
        if key not in shown_keys:
            if len(shown_keys) >= n_hands:
                break
            shown_keys.append(key)
            print(f"\n===== game {row['game_id']} hand {row['hand_id']} =====")

        bits = [f"p{row['player_id']}", row["phase"], f"trick={row['trick_num']}"]
        if row["trump_suit"]:
            bits.append(f"trump={row['trump_suit']}")
        print(
            f"  [{row['decision_id']:>2}] {' '.join(bits):<28} "
            f"hand={row['hand_cards']} legal={row['legal_actions']} "
            f"-> {row['action_taken']}"
            + (f"  reward={row['reward']} done" if row["done"] else "")
        )


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect and parse 500 card game data")
    parser.add_argument("--games", type=int, default=100, help="Number of games to run")
    parser.add_argument("--games-out", type=str, default="data/games.jsonl", help="Output games JSONL file")
    parser.add_argument("--decisions-out", type=str, default="data/decisions.jsonl", help="Output decisions JSONL file")
    parser.add_argument("--server", type=str, default="./500/server", help="Path to server binary")
    parser.add_argument("--port", type=int, default=8080, help="Port for server")
    parser.add_argument("--password", type=str, default="pass", help="Server password")
    parser.add_argument("--bots", type=str, default="2222", help="Bot config e.g. 2222")
    parser.add_argument("--keep-raw", action="store_true", help="Save raw .txt files to data/raw/")
    parser.add_argument("--review", type=int, default=0, help="Pretty-print N hands of decisions for manual QA")
    args = parser.parse_args()

    raw_dir = "data/raw" if args.keep_raw else None

    raw_results = run_games(
        n=args.games,
        server_path=args.server,
        port=args.port,
        password=args.password,
        bot_types=args.bots,
        raw_dir=raw_dir
    )

    games_rows = []
    decisions_rows = []
    total_skipped = 0
    for game_id, (raw_text, timestamp) in enumerate(raw_results, start=1):
        games_row, decisions, skipped = process_game(game_id, raw_text, timestamp)
        games_rows.append(games_row)
        decisions_rows.extend(decisions)
        total_skipped += skipped

    write_jsonl(args.games_out, games_rows)
    write_jsonl(args.decisions_out, decisions_rows)

    print(f"  games: {len(games_rows)} rows -> {args.games_out}")
    print(f"  decisions: {len(decisions_rows)} rows -> {args.decisions_out}")
    if total_skipped:
        print(f"  ({total_skipped} hand(s) skipped: redealt or unreconstructable - see module docstring)")

    if args.review:
        print_review(decisions_rows, args.review)


if __name__ == "__main__":
    main()
