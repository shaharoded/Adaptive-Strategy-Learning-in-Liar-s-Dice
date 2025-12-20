
"""
config.py
Defines the GameConfig dataclass, which centralizes all rule options and numeric constraints for the Liar's Dice engine.
Related modules:
- engine.py: Uses GameConfig to initialize and enforce game rules.
- bid.py: Uses GameConfig for bid validation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class GameConfig:
    """
    Centralizes all rule options and numeric constraints for a Liar's Dice game.
    Fields:
        num_players (int): Number of players (default 2).
        total_dice (int): Total dice (used if dice_distribution is None).
        dice_distribution (tuple): Dice per player (overrides total_dice).
        faces (tuple): Allowed die faces.
        ones_wild (bool): If True, ones are wild.
        bid_ordering (str): Bid ordering rule.
        allow_opening_bid_constraints (bool): Extra constraints for opening bid.
        max_turns (int): Max turns per round.
        rng_seed (int|None): Seed for deterministic games.
    """
    num_players: int = 2
    total_dice: int = 10
    # if dice_distribution is None, engine will interpret total_dice as per-player dice count
    dice_distribution: Optional[Tuple[int, ...]] = None
    faces: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    ones_wild: bool = False
    bid_ordering: str = "quantity_then_face"
    allow_opening_bid_constraints: bool = False
    max_turns: int = 64
    rng_seed: Optional[int] = 69
