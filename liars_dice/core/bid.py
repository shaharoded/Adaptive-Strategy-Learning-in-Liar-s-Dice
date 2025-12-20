
"""
bid.py
Defines the Bid model for Liar's Dice, including validation and ordering logic.
Related modules:
- actions.py: Uses Bid in BidAction.
- engine.py: Validates and compares bids to enforce game rules.
"""

from dataclasses import dataclass
from typing import Any



@dataclass(frozen=True)
class Bid:
    """
    Represents a bid in Liar's Dice: a claim about the quantity and face value of dice.
    Args:
        quantity (int): Number of dice claimed.
        face (int): Face value claimed (1-6).
    """
    quantity: int
    face: int

    def validate(self, config: Any) -> None:
        """
        Validates the bid against game configuration.
        Args:
            config: GameConfig or similar with dice distribution and rules.
        Raises:
            ValueError: If bid is out of bounds.
        """
        if not (1 <= self.face <= 6):
            raise ValueError("face must be between 1 and 6")
        # Determine the maximum possible dice in the game. Prefer explicit dice_distribution if provided.
        if hasattr(config, "dice_distribution") and config.dice_distribution:
            max_total = sum(config.dice_distribution)
        else:
            # interpret config.total_dice as per-player count
            max_total = getattr(config, "total_dice", 0) * getattr(config, "num_players", 1)
        if not (1 <= self.quantity <= max_total):
            raise ValueError("quantity must be between 1 and total_dice")

    def is_higher_than(self, other: 'Bid') -> bool:
        """
        Checks if this bid is strictly higher than another bid, per game rules.
        Args:
            other (Bid): The previous bid to compare against (or None).
        Returns:
            bool: True if this bid is higher, False otherwise.
        """
        if other is None:
            return True
        # Default ordering: quantity then face
        if self.quantity != other.quantity:
            return self.quantity > other.quantity
        return self.face > other.face

