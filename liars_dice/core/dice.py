import random
from typing import List


def roll_die(rng: random.Random) -> int:
    return rng.randint(1, 6)


def roll_n(n: int, rng: random.Random) -> List[int]:
    return [roll_die(rng) for _ in range(n)]

