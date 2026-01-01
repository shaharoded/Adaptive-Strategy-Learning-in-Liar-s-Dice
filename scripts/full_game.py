"""full_game.py
Simulate full Liar's Dice matches between two agents. Each match starts with a fixed
number of dice per player (default 5). Players play rounds; after each round the
loser loses one die. Rounds repeat until one player has zero dice and therefore
loses the full match.

This script reuses the per-round game loop from run_experiments.py but wraps
it into a match loop that decrements dice counts after each round.

Usage: edit the configuration in main() and run this script.
"""
import os
import datetime
import hashlib
from typing import Any, Dict, List, Tuple

from liars_dice.persistence import csv_io
from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.reward import get_reward
from liars_dice.agents import AGENT_MAP


def generate_match_id(agent0_cls, agent1_cls, timestamp):
    raw = f"{timestamp}_{agent0_cls.__name__}_{agent1_cls.__name__}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_full_match(agent0_cls, agent1_cls, cfg: GameConfig, match_index: int, match_id: str, timestamp: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a full match: multiple rounds until one player has zero dice.

    Returns a tuple of (match_summary, trajectory_rows) similar in structure to
    run_experiments.run_game but aggregated across rounds.
    """
    # initialize engine and agents
    engine = GameEngine(cfg)
    a0 = agent0_cls()
    a1 = agent1_cls()

    # per-match accumulators
    error = None
    total_rounds = 0
    total_steps = 0
    total_bids = 0
    total_calls = 0
    total_bluffs_called = 0
    end_reason = None

    trajectory_rows: List[Dict[str, Any]] = []

    # Play rounds until one player has zero dice
    try:
        while True:
            # If any player has zero dice, match ends
            p0, p1 = engine.state.players
            if p0.num_dice <= 0 or p1.num_dice <= 0:
                break

            # start a new round (roll dice according to current num_dice)
            engine.start_new_round()
            total_rounds += 1

            # run the round until terminal or max turns
            steps = 0
            max_steps = getattr(cfg, 'max_turns', 1000)
            try:
                while not engine.is_terminal() and steps < max_steps:
                    current = engine.state.public.current_player
                    view = engine.get_view(current)
                    agent = a0 if current == 0 else a1
                    action = agent.choose_action(view)
                    state_repr = str(view)
                    action_repr = str(action)
                    engine.apply_action(current, action)
                    steps += 1
                    # collect events emitted by engine
                    popped = engine.pop_events()
                    for ev in popped:
                        t = ev.get("type")
                        if t == "BidPlaced":
                            total_bids += 1
                        elif t == "LiarCalled":
                            total_calls += 1
                        elif t == "RoundEnded":
                            was_true = ev.get("was_true")
                            if was_true is False:
                                total_bluffs_called += 1
                        # normalize event type to string to satisfy type-checkers
                        t_str = t if t is not None else "Unknown"
                        r = get_reward(t_str, view, action, current, engine.state.public)
                        trajectory_rows.append({
                            "game_id": match_id,
                            "round": engine.state.public.round_index,
                            "event_type": t_str,
                            "turn_index": engine.state.public.turn_index,
                            "player": ev.get("player", current),
                            "player_type": agent.__class__.__name__,
                            "payload": str(ev.get("bid", ev)),
                            "timestamp": timestamp,
                            "state": state_repr,
                            "action": action_repr,
                            "reward": r,
                        })
            except IllegalMoveError as e:
                # illegal move ends the round and is recorded
                error = str(e)
                end_reason = "IllegalMoveError"
                current = engine.state.public.current_player
                r = get_reward("Error", engine.get_view(current), "Error", current, engine.state.public)
                trajectory_rows.append({
                    "game_id": match_id,
                    "round": engine.state.public.round_index,
                    "event_type": "Error",
                    "turn_index": engine.state.public.turn_index,
                    "player": current,
                    "player_type": (a0 if current == 0 else a1).__class__.__name__,
                    "payload": str(e),
                    "timestamp": timestamp,
                    "state": str(engine.get_view(current)),
                    "action": "Error",
                    "reward": r,
                })
                engine.state.public.status = "ENDED"
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error = f"UnexpectedException: {e}"
                end_reason = "UnexpectedException"
                current = engine.state.public.current_player
                r = get_reward("Error", engine.get_view(current), "Error", current, engine.state.public)
                trajectory_rows.append({
                    "game_id": match_id,
                    "round": engine.state.public.round_index,
                    "event_type": "Error",
                    "turn_index": engine.state.public.turn_index,
                    "player": current,
                    "player_type": (a0 if current == 0 else a1).__class__.__name__,
                    "payload": tb,
                    "timestamp": timestamp,
                    "state": str(engine.get_view(current)),
                    "action": "Error",
                    "reward": r,
                })
                engine.state.public.status = "ENDED"

            # collect final events from round (if any)
            final = engine.pop_events()
            for ev in final:
                t = ev.get("type")
                # For final event use the player's view if possible (fall back to player 0)
                player_for_view = ev.get("player", 0)
                t_str = t if t is not None else "Unknown"
                r = get_reward(t_str, engine.get_view(player_for_view), "FinalEvent", player_for_view, engine.state.public)
                trajectory_rows.append({
                    "game_id": match_id,
                    "round": engine.state.public.round_index,
                    "event_type": t_str,
                    "turn_index": engine.state.public.turn_index,
                    "player": ev.get("player", None),
                    "player_type": None,
                    "payload": str(ev.get("bid", ev)),
                    "timestamp": timestamp,
                    "state": str(engine.get_view(player_for_view)),
                    "action": "FinalEvent",
                    "reward": r,
                })

            # Update aggregate counters
            total_steps += steps

            # If round ended with a winner/loser, decrement loser's dice and continue match
            if engine.state.public.winner is not None:
                loser = engine.state.public.loser
                # decrement loser's dice but not below zero
                engine.state.players[loser].num_dice = max(0, engine.state.players[loser].num_dice - 1)
                # record the dice loss as an event in the trajectory for clarity
                trajectory_rows.append({
                    "game_id": match_id,
                    "round": engine.state.public.round_index,
                    "event_type": "DiceLost",
                    "turn_index": engine.state.public.turn_index,
                    "player": loser,
                    "player_type": None,
                    "payload": f"player {loser} lost a die, now has {engine.state.players[loser].num_dice}",
                    "timestamp": timestamp,
                    "state": str(engine.get_view(loser)),
                    "action": "DiceLost",
                    "reward": 0,
                })

            # safety: if rounds exceed a large number, break to avoid infinite
            if total_rounds > 1000:
                end_reason = end_reason or "match_round_limit_reached"
                break

        # Match ended because one player reached zero dice
        p0, p1 = engine.state.players
        if p0.num_dice <= 0:
            match_winner = 1
            match_loser = 0
        elif p1.num_dice <= 0:
            match_winner = 0
            match_loser = 1
        else:
            match_winner = None
            match_loser = None
            if end_reason is None:
                end_reason = "unknown"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error = f"UnexpectedExceptionDuringMatch: {e}"
        end_reason = "UnexpectedExceptionDuringMatch"
        trajectory_rows.append({
            "game_id": match_id,
            "round": engine.state.public.round_index,
            "event_type": "Error",
            "turn_index": engine.state.public.turn_index,
            "player": None,
            "player_type": None,
            "payload": tb,
            "timestamp": timestamp,
            "state": str(engine.get_view(0)),
            "action": "Error",
            "reward": 0,
        })
        match_winner = None
        match_loser = None

    summary_row = {
        "game_id": match_id,
        "game_index": match_index,
        "timestamp": timestamp,
        "agent0": agent0_cls.__name__,
        "agent1": agent1_cls.__name__,
        "winner": match_winner,
        "loser": match_loser,
        "starting_dice_per_player": cfg.total_dice,
        "rounds_played": total_rounds,
        "steps": total_steps,
        "bids": total_bids,
        "calls": total_calls,
        "bluffs_called": total_bluffs_called,
        "error": error,
        "end_reason": end_reason,
    }

    return summary_row, trajectory_rows


def main():
    #################################
    #        Configuration
    ##################################
    agent_1 = "random"
    agent_2 = "random"
    number_of_matches = 10
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    summary_csv = os.path.join(data_dir, "match_summary.csv")
    trajectory_csv = os.path.join(data_dir, "match_trajectory.csv")

    if agent_1 not in AGENT_MAP or agent_2 not in AGENT_MAP:
        raise SystemExit(f"Unknown agent. Supported: {list(AGENT_MAP.keys())}")

    agent0_cls = AGENT_MAP[agent_1]
    agent1_cls = AGENT_MAP[agent_2]
    # default GameConfig uses total_dice as per-player starting dice
    cfg = GameConfig()

    summary_header = csv_io.get_summary_header()
    trajectory_header = csv_io.get_trajectory_header()

    for i in range(number_of_matches):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        match_id = generate_match_id(agent0_cls, agent1_cls, f"{timestamp}_{i}")
        print(f"Running match {i+1}/{number_of_matches}...", end=" ")
        summary_row, trajectory_rows = run_full_match(agent0_cls, agent1_cls, cfg, i, match_id, timestamp)
        csv_io.append_row_to_csv(summary_row, summary_csv, summary_header)
        csv_io.append_rows_to_csv(trajectory_rows, trajectory_csv, trajectory_header)
        print("done")

    print(f"All matches finished. Data saved to {data_dir}/match_summary.csv and {data_dir}/match_trajectory.csv")


if __name__ == "__main__":
    main()
