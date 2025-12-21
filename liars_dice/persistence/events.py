
"""
events.py
Defines the GameEvent dataclass for event-sourced recording of game actions and state changes.
Used by recorder.py and engine.py to log all important game events for replay, analysis, or persistence.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GameEvent:
    """
    Represents a single event in the game (e.g., dice rolled, bid placed, round ended).
    Fields:
        game_id (str): Unique game identifier.
        event_type (str): Type of event (e.g., 'BidPlaced').
        payload (dict): Event-specific data.
        player_type (str|None): Agent class name or 'Human' for the player who took the action (optional, for moves/actions).
    """
    game_id: str
    event_type: str
    payload: Dict[str, Any]
    player_type: str = None

