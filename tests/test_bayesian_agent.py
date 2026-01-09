import unittest

from liars_dice.agents.bayesian_agent import BayesianAgent, BayesianSettings
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid
from liars_dice.core.config import GameConfig


class TestBayesianAgent(unittest.TestCase):
    def _make_view(self, *, my_dice, last_bid, dice_counts=(5, 5), ones_wild=False, turn_index=0):
        config = GameConfig(dice_distribution=tuple(dice_counts), ones_wild=ones_wild)

        class PublicView:
            def __init__(self):
                self.dice_counts = dice_counts
                self.last_bid = last_bid
                self.turn_index = turn_index
                self.current_player = 0
                self.bid_history = []
                self.status = "BIDDING"

        return {"public": PublicView(), "my_dice": tuple(my_dice), "config": config}

    def test_calls_liar_when_bid_is_very_unlikely(self):
        # With 10 dice total, bidding 10 sixes is extremely unlikely unless we already hold many sixes.
        view = self._make_view(my_dice=[1, 2, 3, 4, 5], last_bid=Bid(10, 6), dice_counts=(5, 5))
        agent = BayesianAgent(settings=BayesianSettings(base_call_threshold=0.25))
        action = agent.choose_action(view)
        self.assertIsInstance(action, CallLiarAction)

    def test_raises_minimally_when_bid_is_plausible(self):
        view = self._make_view(my_dice=[3, 3, 4, 5, 6], last_bid=Bid(2, 3), dice_counts=(5, 5))
        agent = BayesianAgent(settings=BayesianSettings(base_call_threshold=0.05, min_raise_truth_prob=0.0))
        action = agent.choose_action(view)
        self.assertIsInstance(action, BidAction)
        self.assertTrue(action.bid.is_higher_than(Bid(2, 3)))

    def test_ones_wild_makes_non_one_bids_more_plausible(self):
        # If ones are wild, a bid on 6's effectively counts 6s or 1s.
        view_wild = self._make_view(my_dice=[1, 2, 3, 4, 5], last_bid=Bid(6, 6), dice_counts=(5, 5), ones_wild=True)
        view_plain = self._make_view(my_dice=[1, 2, 3, 4, 5], last_bid=Bid(6, 6), dice_counts=(5, 5), ones_wild=False)

        agent = BayesianAgent(settings=BayesianSettings(base_call_threshold=0.2))
        p_wild = agent._truth_prob_for_bid(view_wild, view_wild["public"].last_bid)
        p_plain = agent._truth_prob_for_bid(view_plain, view_plain["public"].last_bid)
        self.assertGreater(p_wild, p_plain)


if __name__ == "__main__":
    unittest.main()

