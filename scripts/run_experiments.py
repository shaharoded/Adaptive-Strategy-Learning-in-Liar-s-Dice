import os
import json
import datetime
from typing import Any, Dict

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.persistence import serializer


# Import the central agent registry
from liars_dice.agents import AGENT_MAP


def run_game(agent0_cls, agent1_cls, cfg: GameConfig, game_index: int) -> Dict[str, Any]:
    """
    Run a single game between two agent classes with the given configuration.
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

    game_events = []
    error = None
    # per-game counters for statistics
    steps = 0
    bids = 0
    calls = 0
    bluffs_called = 0
    end_reason = None
    # safety guard to avoid infinite games
    max_steps = getattr(cfg, 'max_turns', 1000)
    try:
        while not engine.is_terminal() and steps < max_steps:
            current = engine.state.public.current_player
            view = engine.get_view(current)
            agent = a0 if current == 0 else a1
            action = agent.choose_action(view)
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
            game_events.extend(popped)

    except IllegalMoveError as e:
        # expected illegal moves coming from GameEngine (bad action / turn / no bid to call)
        error = str(e)
        end_reason = "IllegalMoveError"
        game_events.append({"type": "Error", "error": error, "kind": end_reason})
        # mark ended (no valid winner in this case)
        engine.state.public.status = "ENDED"
    except Exception as e:
        # catch any unexpected exception, record traceback for debugging
        import traceback

        tb = traceback.format_exc()
        error = f"UnexpectedException: {e}"
        end_reason = "UnexpectedException"
        game_events.append({"type": "Error", "error": error, "traceback": tb})
        engine.state.public.status = "ENDED"

    # collect final events from engine (if any)
    final = engine.pop_events()
    if final:
        game_events.extend(final)

    # if we exited because of reaching max_steps, mark it
    if not engine.is_terminal() and steps >= max_steps:
        end_reason = end_reason or "max_steps_reached"
        game_events.append({"type": "Error", "error": "max steps reached", "kind": end_reason})
        engine.state.public.status = "ENDED"

    # Build a result dict
    result = {
        "game_index": game_index,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent0": agent0_cls.__name__,
        "agent1": agent1_cls.__name__,
        "config": cfg.__dict__,
        "final_public": engine.state.public.__dict__,
        "players": [p.__dict__ for p in engine.state.players],
        "turn_log": engine.turn_log,
        "events": game_events,
        "error": error,
        "end_reason": end_reason,
        "stats": {
            "steps": steps,
            "bids": bids,
            "calls": calls,
            "bluffs_called": bluffs_called,
            "events_recorded": len(game_events),
        },
    }
    return result


def save_game_result(result: Dict[str, Any], out_dir: str) -> str:
    """
    Save a single game's result dictionary to a JSON file in the output directory.
    Args:
        result (dict): The result dictionary from run_game().
        out_dir (str): Output directory path.
    Returns:
        str: Path to the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = f"game_{result['game_index']:04d}_{result['timestamp'].replace(':', '-')}.json"
    path = os.path.join(out_dir, filename)
    # use serializer.dumps to handle dataclasses
    with open(path, "w", encoding="utf-8") as f:
        f.write(serializer.dumps(result))
    return path


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
    results_output_dir = "results" # output directory

    if agent_1 not in AGENT_MAP or agent_2 not in AGENT_MAP:
        raise SystemExit(f"Unknown agent. Supported: {list(AGENT_MAP.keys())}")

    agent0_cls = AGENT_MAP[agent_1]
    agent1_cls = AGENT_MAP[agent_2]

    cfg = GameConfig()

    # cumulative aggregation for averages
    total_steps = 0
    total_bids = 0
    total_calls = 0
    total_bluffs = 0

    summary = {"total": number_of_games, "wins_agent0": 0, "wins_agent1": 0, "errors": 0, "files": [], "per_game_stats": []}

    for i in range(number_of_games):
        print(f"Running game {i+1}/{number_of_games}...", end=" ")
        result = run_game(agent0_cls, agent1_cls, cfg, i)
        path = save_game_result(result, results_output_dir)
        summary["files"].append(path)
        # record per-game stats in summary list
        stats = result.get("stats", {})
        summary["per_game_stats"].append({"game_index": i, **stats})
        total_steps += stats.get("steps", 0)
        total_bids += stats.get("bids", 0)
        total_calls += stats.get("calls", 0)
        total_bluffs += stats.get("bluffs_called", 0)
        # determine winner from final_public
        winner = result.get("final_public", {}).get("winner")
        if winner is None:
            if result.get("error"):
                summary["errors"] += 1
        else:
            if winner == 0:
                summary["wins_agent0"] += 1
            elif winner == 1:
                summary["wins_agent1"] += 1
        print("done ->", os.path.basename(path))

    # compute averages
    if number_of_games > 0:
        summary["avg_steps"] = total_steps / number_of_games
        summary["avg_bids"] = total_bids / number_of_games
        summary["avg_calls"] = total_calls / number_of_games
        summary["avg_bluffs_called"] = total_bluffs / number_of_games
    else:
        summary["avg_steps"] = 0
        summary["avg_bids"] = 0
        summary["avg_calls"] = 0
        summary["avg_bluffs_called"] = 0

    # write summary
    summary_path = os.path.join(results_output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))

    print("All games finished. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
