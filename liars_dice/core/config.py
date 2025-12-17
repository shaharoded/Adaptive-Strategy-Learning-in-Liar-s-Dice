from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class GameConfig:
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
