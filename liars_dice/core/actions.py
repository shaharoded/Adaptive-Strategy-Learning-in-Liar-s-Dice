from dataclasses import dataclass
from .bid import Bid
from typing import Any


class Action:
    pass


@dataclass(frozen=True)
class BidAction(Action):
    bid: Bid


@dataclass(frozen=True)
class CallLiarAction(Action):
    pass

