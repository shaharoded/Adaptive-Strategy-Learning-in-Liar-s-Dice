import os
import datetime
import hashlib
from typing import Any, Dict

# Use persistence CSV utilities
from liars_dice.persistence import csv_io

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.reward import get_reward

# Import the central agent registry
from liars_dice.agents import AGENT_MAP


def generate_game_id(agent0_cls, agent1_cls, timestamp):
    raw = f"{timestamp}_{agent0_cls.__name__}_{agent1_cls.__name__}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def run_game(agent0_cls, agent1_cls, cfg: GameConfig, game_index: int, game_id: str, timestamp: str) -> Dict[str, Any]:
    """
    Run a single game (single round) between two agent classes with the given configuration.
    Each game ends when a winner is declared or an error occurs.
    Args:
        agent0_cls: Class of agent 0 (must implement choose_action(view)).
        agent1_cls: Class of agent 1.
        cfg (GameConfig): Game configuration.
        game_index (int): Index of the game (for output naming).
    Returns:
        dict: Result dictionary with game data, events, stats, and errors if any.
    """
    engine = GameEngine(cfg)
    # instantiate agents; give each a seeded RNG for reproducibility
    a0 = agent0_cls()
    a1 = agent1_cls()

    engine.start_new_round()
    error = None
    # per-game counters for statistics
    steps = 0
    bids = 0
    calls = 0
    bluffs_called = 0
    end_reason = None
    # safety guard to avoid infinite games
    max_steps = getattr(cfg, 'max_turns', 1000)
    trajectory_rows = []
    try:
        while not engine.is_terminal() and steps < max_steps:
            current = engine.state.public.current_player
            view = engine.get_view(current)
            agent = a0 if current == 0 else a1
            action = agent.choose_action(view)
            # Serialize state and action for ML/RL
            state_repr = str(view)
            action_repr = str(action)
            # Apply action
            engine.apply_action(current, action)
            steps += 1
            # capture incremental events and update stats
            popped = engine.pop_events()
            for ev in popped:
                t = ev.get("type")
                if t == "BidPlaced":
                    bids += 1
                elif t == "LiarCalled":
                    calls += 1
                elif t == "RoundEnded":
                    # if was_true == False then the caller correctly identified a bluff
                    was_true = ev.get("was_true")
                    if was_true is False:
                        bluffs_called += 1
                r = get_reward(t, view, action, current, engine.state.public)
                trajectory_rows.append({
                    "game_id": game_id,
                    "event_type": t,
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
        # expected illegal moves coming from GameEngine (bad action / turn / no bid to call)
        error = str(e)
        end_reason = "IllegalMoveError"
        r = get_reward("Error", engine.get_view(current), "Error", current, engine.state.public)
        trajectory_rows.append({
            "game_id": game_id,
            "event_type": "Error",
            "turn_index": engine.state.public.turn_index,
            "player": current,
            "player_type": agent.__class__.__name__,
            "payload": str(e),
            "timestamp": timestamp,
            "state": str(engine.get_view(current)),
            "action": "Error",
            "reward": r,
        })
        engine.state.public.status = "ENDED"
    except Exception as e:
        # catch any unexpected exception, record traceback for debugging
        import traceback
        tb = traceback.format_exc()
        error = f"UnexpectedException: {e}"
        end_reason = "UnexpectedException"
        r = get_reward("Error", engine.get_view(current), "Error", current, engine.state.public)
        trajectory_rows.append({
            "game_id": game_id,
            "event_type": "Error",
            "turn_index": engine.state.public.turn_index,
            "player": current,
            "player_type": agent.__class__.__name__,
            "payload": tb,
            "timestamp": timestamp,
            "state": str(engine.get_view(current)),
            "action": "Error",
            "reward": r,
        })
        engine.state.public.status = "ENDED"

    # collect final events from engine (if any)
    final = engine.pop_events()
    for ev in final:
        t = ev.get("type")
        r = get_reward(t, engine.get_view(ev.get("player", 0)), "FinalEvent", ev.get("player", 0), engine.state.public)
        trajectory_rows.append({
            "game_id": game_id,
            "event_type": t,
            "turn_index": engine.state.public.turn_index,
            "player": ev.get("player", None),
            "player_type": None,
            "payload": str(ev.get("bid", ev)),
            "timestamp": timestamp,
            "state": str(engine.get_view(ev.get("player", 0))),
            "action": "FinalEvent",
            "reward": r,
        })
    if not engine.is_terminal() and steps >= max_steps:
        end_reason = end_reason or "max_steps_reached"
        trajectory_rows.append({
            "game_id": game_id,
            "event_type": "Error",
            "turn_index": engine.state.public.turn_index,
            "player": None,
            "player_type": None,
            "payload": "max steps reached",
            "timestamp": timestamp,
            "state": str(engine.get_view(0)),
            "action": "Error",
            "reward": get_reward("Error", engine.get_view(0), "Error", 0, engine.state.public),
        })
        engine.state.public.status = "ENDED"
    # If game ended normally (winner declared), set end_reason
    if end_reason is None and engine.state.public.winner is not None:
        end_reason = "winner declared"
    summary_row = {
        "game_id": game_id,
        "game_index": game_index,
        "timestamp": timestamp,
        "agent0": agent0_cls.__name__,
        "agent1": agent1_cls.__name__,
        "winner": engine.state.public.winner,
        "loser": engine.state.public.loser,
        "steps": steps,
        "bids": bids,
        "calls": calls,
        "bluffs_called": bluffs_called,
        "error": error,
        "end_reason": end_reason,
    }
    return summary_row, trajectory_rows





"""
Run a batch of Liar's Dice games between two agents, save results, and print a summary.
Edit the configuration section to change agents, number of games, or output directory.
"""
def main():
    #################################
    #        Configuration
    ##################################
    agent_1 = "random" # agent name
    agent_2 = "random" # agent name
    number_of_games = 10
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    summary_csv = os.path.join(data_dir, "game_summary.csv")
    trajectory_csv = os.path.join(data_dir, "game_trajectory.csv")

    if agent_1 not in AGENT_MAP or agent_2 not in AGENT_MAP:
        raise SystemExit(f"Unknown agent. Supported: {list(AGENT_MAP.keys())}")

    agent0_cls = AGENT_MAP[agent_1]
    agent1_cls = AGENT_MAP[agent_2]
    cfg = GameConfig()

    summary_header = csv_io.get_summary_header()
    trajectory_header = csv_io.get_trajectory_header()

    for i in range(number_of_games):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        game_id = generate_game_id(agent0_cls, agent1_cls, f"{timestamp}_{i}")
        print(f"Running game {i+1}/{number_of_games}...", end=" ")
        summary_row, trajectory_rows = run_game(agent0_cls, agent1_cls, cfg, i, game_id, timestamp)
        csv_io.append_row_to_csv(summary_row, summary_csv, summary_header)
        csv_io.append_rows_to_csv(trajectory_rows, trajectory_csv, trajectory_header)
        print("done")

    print(f"All games finished. Data saved to {data_dir}/game_summary.csv and {data_dir}/game_trajectory.csv")


if __name__ == "__main__":
    main()
