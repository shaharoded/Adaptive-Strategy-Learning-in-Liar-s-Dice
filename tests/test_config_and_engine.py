import unittest
from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine


class TestConfigAndEngine(unittest.TestCase):
    """
    Tests around how `GameConfig` is interpreted by the engine with respect to per-player
    dice distribution. These tests verify that:
      - By default each player receives `total_dice` (5) if `dice_distribution` is not set.
      - Changing `total_dice` affects all players (applies to each player equally).
      - Providing an explicit `dice_distribution` overrides `total_dice`.
    """

    def test_default_per_player_dice(self):
        cfg = GameConfig(total_dice=5, dice_distribution=None)  # default total_dice=5, dice_distribution=None
        engine = GameEngine(cfg)
        p0, p1 = engine.state.players
        self.assertEqual(p0.num_dice, 5)
        self.assertEqual(p1.num_dice, 5)

    def test_total_dice_affects_both_players(self):
        cfg = GameConfig(total_dice=3)
        engine = GameEngine(cfg)
        p0, p1 = engine.state.players
        # both players should now have 3 dice each
        self.assertEqual(p0.num_dice, 3)
        self.assertEqual(p1.num_dice, 3)

    def test_explicit_distribution_overrides(self):
        cfg = GameConfig(dice_distribution=(4, 6))
        engine = GameEngine(cfg)
        p0, p1 = engine.state.players
        self.assertEqual(p0.num_dice, 4)
        self.assertEqual(p1.num_dice, 6)


if __name__ == '__main__':
    unittest.main()

