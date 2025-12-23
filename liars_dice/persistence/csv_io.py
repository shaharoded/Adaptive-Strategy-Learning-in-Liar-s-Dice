"""
csv_io.py
Persistence utilities for writing Liar's Dice game summary and trajectory data to CSV files.
"""

import os
import csv
from typing import Dict, List, Any

SUMMARY_HEADER = [
    "game_id", "game_index", "timestamp", "agent0", "agent1", "winner", "loser",
    "steps", "bids", "calls", "bluffs_called", "error", "end_reason",
    # Added fields used by full_game.py for match-level reporting
    "starting_dice_per_player", "rounds_played",
]
TRAJECTORY_HEADER = [
    "game_id", "round", "event_type", "turn_index", "player", "player_type",
    "payload", "timestamp", "state", "action", "reward"
]

def append_row_to_csv(row: Dict[str, Any], csv_path: str, header: List[str]):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def append_rows_to_csv(rows: List[Dict[str, Any]], csv_path: str, header: List[str]):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

def get_summary_header():
    return SUMMARY_HEADER.copy()

def get_trajectory_header():
    return TRAJECTORY_HEADER.copy()
