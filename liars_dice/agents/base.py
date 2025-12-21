from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """
    Abstract base class for all Liar's Dice agents.
    Agents must implement choose_action(view), which receives a player-specific view of the game state and returns an Action.
    Common agent utilities can be added here for reuse.
    """

    @abstractmethod
    def choose_action(self, view: Any):
        """
        Given a player-specific view, return the next Action to take.
        Args:
            view (dict): Player view with keys 'public', 'my_dice', and optionally 'config'.
        Returns:
            Action: The action to take (BidAction or CallLiarAction).
        """
        raise NotImplementedError

    def my_count_of_face(self, my_dice, face: int) -> int:
        """
        Count how many dice of a given face the agent holds.
        Args:
            my_dice (iterable): The agent's private dice.
            face (int): The face value to count.
        Returns:
            int: Number of dice showing the given face.
        """
        return sum(1 for d in my_dice if d == face)

    def call_liar_deterministic(self, my_dice, last_bid, estimated_total):
        """
        Determine if the agent should call liar with certainty, given the last bid, own dice, and total dice in play.
        Returns True if even with all possible opponent dice, the bid cannot be true.
        Args:
            my_dice (iterable): The agent's private dice.
            last_bid (Bid): The last bid made.
            estimated_total (int): Total dice in the game.
        Returns:
            bool: True if the agent should call liar deterministically.
        """
        if last_bid is None:
            return False
        opponent_max = max(0, estimated_total - len(my_dice))
        my_count = self.my_count_of_face(my_dice, last_bid.face)
        return my_count + opponent_max < last_bid.quantity
