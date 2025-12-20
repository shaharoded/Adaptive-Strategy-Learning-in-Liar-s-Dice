import unittest
from liars_dice.core.bid import Bid
from liars_dice.core.config import GameConfig


class TestBidValidationAndOrdering(unittest.TestCase):
    def test_valid_bid_within_defaults(self):
        cfg = GameConfig()  # defaults: num_players=2, total_dice=10 -> max_total = 10*2
        b = Bid(3, 4)
        # should not raise
        b.validate(cfg)

    def test_invalid_face_low_and_high(self):
        cfg = GameConfig()
        with self.assertRaises(ValueError):
            Bid(1, 0).validate(cfg)
        with self.assertRaises(ValueError):
            Bid(1, 7).validate(cfg)

    def test_invalid_quantity_too_high_and_low(self):
        cfg = GameConfig(total_dice=2, num_players=2)
        with self.assertRaises(ValueError):
            Bid(0, 1).validate(cfg)
        with self.assertRaises(ValueError):
            Bid(5, 1).validate(cfg)

    def test_validate_with_explicit_dice_distribution(self):
        cfg = GameConfig(dice_distribution=(2, 1))
        Bid(3, 6).validate(cfg)  # sum = 3
        with self.assertRaises(ValueError):
            Bid(4, 1).validate(cfg)

    def test_is_higher_than_none_and_comparisons(self):
        cfg = GameConfig()
        b = Bid(2, 3)
        self.assertTrue(b.is_higher_than(None))
        self.assertTrue(Bid(3, 1).is_higher_than(Bid(2, 6)))
        self.assertTrue(Bid(2, 4).is_higher_than(Bid(2, 3)))
        self.assertFalse(Bid(2, 3).is_higher_than(Bid(2, 3)))


if __name__ == '__main__':
    unittest.main()

