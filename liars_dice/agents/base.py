from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def choose_action(self, view: Any):
        raise NotImplementedError

