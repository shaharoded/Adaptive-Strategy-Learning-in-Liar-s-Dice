from abc import ABC, abstractmethod
from typing import Any

# --- Approximate Nash/CFR Agent (stub) ---
class UntrainedAgentException(Exception):
    """Raised when a NashCFRAgent/PPOAgent is used without a trained policy file present."""
    pass

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

    def is_bid_universally_impossible(self, bid, total_dice, config=None):
        """
        Check if a bid is universally impossible (no one can make it).
        A bid is universally impossible if its quantity exceeds total dice in play.
        
        This blocks our own bid suggestions (never suggest quantity > total_dice),
        which can immediately be called as liar by the opponent.
        
        Args:
            bid (Bid): The bid to check.
            total_dice (int): Total dice in the game.
            config (Config): Game config object (used to check ones_wild setting).
            
        Returns:
            bool: True if the bid is universally impossible.
        
        NOTE: This function takes config as an argument for future extensibility,
        """
        if bid is None:
            return False
        return bid.quantity > total_dice
    
    def is_opponent_bid_provably_false(self, opponent_bid, my_dice, total_dice, config=None):
        """
        Check if opponent's bid is mathematically impossible given our knowledge.
        
        This is used to automatically call liar when we can PROVE the opponent is lying
        based on our dice and the total dice count. When ones_wild=True, ones count as
        wildcards for any face, making more bids possible.
        
        Example: if we have all 6s and they bid (3, 1), we know they can have at most 
        (total_dice - len(my_dice)) ones. With ones_wild=True, ones also count as 6s,
        so their ability to make this bid depends on how many ones they could have.
        
        Note: This does NOT apply to our own bids - we CAN bluff about faces we don't have,
        as long as the quantity is <= total_dice.

        This function is harsher than is_bid_universally_impossible, 
        which only checks if quantity > total_dice and designed to prevent us from making transparent bluffs.
        
        Args:
            opponent_bid (Bid): The bid made by opponent.
            my_dice (iterable): Our private dice.
            total_dice (int): Total dice in the game.
            config (Config): Game config object (used to check ones_wild setting).
            
        Returns:
            bool: True if opponent's bid is provably false.
        """
        if opponent_bid is None:
            return False
        # If quantity > total, it's universally impossible
        if opponent_bid.quantity > total_dice:
            return True
        # Check if we can mathematically prove they can't have this many of this face
        opponent_max = max(0, total_dice - len(my_dice))
        my_count = self.my_count_of_face(my_dice, opponent_bid.face)
        
        # If ones are wild and the bid is NOT for ones, ones also count as matches
        ones_wild = getattr(config, 'ones_wild', False) if config else False
        if ones_wild and opponent_bid.face != 1:
            my_count += self.my_count_of_face(my_dice, 1)
        
        return my_count + opponent_max < opponent_bid.quantity
