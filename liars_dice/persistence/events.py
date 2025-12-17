from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GameEvent:
    game_id: str
    event_type: str
    payload: Dict[str, Any]

