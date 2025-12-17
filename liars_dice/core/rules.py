from typing import Dict, List


def count_matches(all_dice: Dict[int, List[int]], face: int, ones_wild: bool = False) -> int:
    count = 0
    for dice in all_dice.values():
        count += sum(1 for d in dice if d == face)
    if ones_wild and face != 1:
        # add ones as wild
        for dice in all_dice.values():
            count += sum(1 for d in dice if d == 1)
    return count

