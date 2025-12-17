from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Bid:
    quantity: int
    face: int

    def validate(self, config: Any) -> None:
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

    def is_higher_than(self, other: 'Bid', config: Any) -> bool:
        if other is None:
            return True
        # Default ordering: quantity then face
        if self.quantity != other.quantity:
            return self.quantity > other.quantity
        return self.face > other.face

