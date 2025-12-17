import unittest
from liars_dice.core.bid import Bid
from liars_dice.core.config import GameConfig


class TestBidValidation(unittest.TestCase):
    """
    Tests for `Bid.validate` ensuring quantity bounds are enforced relative to the
    game-wide total dice (sum of `dice_distribution` or total_dice * num_players).
    """

    def test_bid_rejects_more_than_total(self):
        cfg = GameConfig(dice_distribution=(2, 3))  # total 5
        b = Bid(6, 2)
        with self.assertRaises(ValueError):
            b.validate(cfg)

    def test_bid_accepts_valid_quantity(self):
        cfg = GameConfig(dice_distribution=(2, 3))
        b = Bid(5, 6)
        # should not raise
        b.validate(cfg)

    def test_bid_uses_total_dice_per_player_when_no_distribution(self):
        cfg = GameConfig(total_dice=4, dice_distribution=None, num_players=2)
        # total dice across players should be 8
        b = Bid(8, 3)
        b.validate(cfg)  # should not raise
        b2 = Bid(9, 3)
        with self.assertRaises(ValueError):
            b2.validate(cfg)


if __name__ == '__main__':
    unittest.main()

