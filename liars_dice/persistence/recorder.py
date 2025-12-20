from typing import List
from .events import GameEvent


class InMemoryRecorder:
    def __init__(self):
        self._events: List[GameEvent] = []

    def record(self, event: GameEvent) -> None:
        self._events.append(event)

    def events(self):
        return list(self._events)

    def flush(self):
        pass

