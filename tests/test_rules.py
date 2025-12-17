import unittest
from liars_dice.core.rules import count_matches


class TestRules(unittest.TestCase):
    def test_count_matches_without_ones_wild(self):
        all_dice = {0: [1, 2], 1: [3, 2, 2]}
        self.assertEqual(count_matches(all_dice, 2, ones_wild=False), 3)
        self.assertEqual(count_matches(all_dice, 1, ones_wild=False), 1)

    def test_count_matches_with_ones_wild(self):
        all_dice = {0: [1, 2], 1: [1, 3, 2]}
        # counting face 2 should include ones
        self.assertEqual(count_matches(all_dice, 2, ones_wild=True), 3)
        # counting face 1 only counts ones
        self.assertEqual(count_matches(all_dice, 1, ones_wild=True), 2)


if __name__ == '__main__':
    unittest.main()

