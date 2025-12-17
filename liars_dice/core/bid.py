from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Bid:
    quantity: int
    face: int

    def validate(self, config: Any) -> None:
        if not (1 <= self.face <= 6):
            raise ValueError("face must be between 1 and 6")
        if not (1 <= self.quantity <= config.total_dice):
            raise ValueError("quantity must be between 1 and total_dice")

    def is_higher_than(self, other: 'Bid', config: Any) -> bool:
        if other is None:
            return True
        # Default ordering: quantity then face
        if self.quantity != other.quantity:
            return self.quantity > other.quantity
        return self.face > other.face

