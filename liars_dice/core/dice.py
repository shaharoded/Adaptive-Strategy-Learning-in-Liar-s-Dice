
"""
dice.py
Defines dice rolling utilities for the Liar's Dice engine.
Related modules:
- engine.py: Uses roll_n to roll dice for each player.
"""

import random
from typing import List


def roll_die(rng: random.Random) -> int:
    """
    Roll a single six-sided die using the provided random number generator.
    Args:
        rng (random.Random): RNG instance.
    Returns:
        int: Die face (1-6).
    """
    return rng.randint(1, 6)


def roll_n(n: int, rng: random.Random) -> List[int]:
    """
    Roll n six-sided dice using the provided RNG.
    Args:
        n (int): Number of dice to roll.
        rng (random.Random): RNG instance.
    Returns:
        list[int]: List of die faces.
    """
    return [roll_die(rng) for _ in range(n)]

