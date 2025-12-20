
"""
actions.py
Defines the base Action type and concrete action classes for the Liar's Dice game engine.
Actions represent moves that players can make (bidding or calling liar).
Related modules:
- bid.py: Defines the Bid model used in BidAction.
- engine.py: Consumes Action objects to update game state.
"""

from dataclasses import dataclass
from .bid import Bid



class Action:
    """
    Base class for all game actions. Subclassed by BidAction and CallLiarAction.
    """
    pass



@dataclass(frozen=True)
class BidAction(Action):
    """
    Represents a bid action: a player claims there are at least 'quantity' dice showing 'face'.
    Args:
        bid (Bid): The bid being placed.
    """
    bid: Bid



@dataclass(frozen=True)
class CallLiarAction(Action):
    """
    Represents the action of calling 'liar' on the previous bid.
    No arguments; triggers a reveal and resolution in the engine.
    """
    pass

