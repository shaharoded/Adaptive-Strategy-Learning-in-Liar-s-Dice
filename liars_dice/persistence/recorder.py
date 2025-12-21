
"""
recorder.py
Implements event recording for Liar's Dice games. Used to store a stream of GameEvent objects for replay, analysis, or persistence.
Can be extended for file/DB storage. InMemoryRecorder is used for tests and in-memory analysis.
Related modules:
- events.py: Defines GameEvent type.
- serializer.py: Used for saving/loading events.
"""

from typing import List
from .events import GameEvent


class InMemoryRecorder:
    """
    Records GameEvent objects in memory for later retrieval.
    Methods:
        record(event): Add a new event.
        events(): Get all recorded events.
        flush(): No-op for in-memory; used in file/DB recorders.
    """
    def __init__(self):
        self._events: List[GameEvent] = []

    def record(self, event: GameEvent) -> None:
        """Add a new event to the recorder."""
        self._events.append(event)

    def events(self):
        """Return all recorded events as a list."""
        return list(self._events)

    def flush(self):
        """No-op for in-memory recorder."""
        pass

