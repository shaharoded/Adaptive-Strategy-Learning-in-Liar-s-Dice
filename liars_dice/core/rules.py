
"""
rules.py
Defines helper functions for Liar's Dice rules, including counting matches and wild ones logic.
Related modules:
- engine.py: Uses count_matches to resolve bids and calls.
"""

from typing import Dict, List

def count_matches(all_dice: Dict[int, List[int]], face: int, ones_wild: bool = False) -> int:
    """
    Count the number of dice matching a given face across all players.
    Args:
        all_dice (dict): Mapping of player_id to list of dice.
        face (int): Face value to count.
        ones_wild (bool): If True, ones count as wild for non-one faces.
    Returns:
        int: Total count of matching dice.
    """
    count = 0
    for dice in all_dice.values():
        count += sum(1 for d in dice if d == face)
    if ones_wild and face != 1:
        # add ones as wild
        for dice in all_dice.values():
            count += sum(1 for d in dice if d == 1)
    return count

