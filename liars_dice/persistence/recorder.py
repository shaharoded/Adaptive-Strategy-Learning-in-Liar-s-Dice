from typing import List, Dict
from .events import GameEvent


class InMemoryRecorder:
    def __init__(self):
        self._events: List[Dict] = []

    def record(self, event: Dict) -> None:
        self._events.append(event)

    def events(self):
        return list(self._events)

    def flush(self):
        pass

