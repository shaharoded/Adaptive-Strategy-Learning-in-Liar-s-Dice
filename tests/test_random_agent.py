import unittest
from liars_dice.agents.random_agent import RandomAgent
from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid


class TestRandomAgent(unittest.TestCase):
    """
    Tests for the `RandomAgent` guard-rails and turn-based call probability:
      - If the last bid is impossible given the agent's private dice and max possible opponent dice,
        the agent must always return a `CallLiarAction`.
      - The agent raises bids within the allowed maximum (does not exceed total dice).
    """

    def test_impossible_bid_calls_liar(self):
        cfg = GameConfig(dice_distribution=(2, 3))
        engine = GameEngine(cfg)
        engine.start_new_round()
        # Force private dice for player 0 to be [1,1] (no face 6)
        engine.state.players[0].private_dice = [1, 1]
        # Create a last bid that asks for 5 sixes — impossible because opponent has at most 3 dice
        engine.state.public.last_bid = Bid(5, 6)
        view = engine.get_view(0)
        agent = RandomAgent(rng=None)
        action = agent.choose_action(view)
        self.assertIsInstance(action, CallLiarAction)

    def test_agent_bid_caps_at_total(self):
        cfg = GameConfig(dice_distribution=(3, 3))
        engine = GameEngine(cfg)
        engine.start_new_round()
        # create a last bid at total 6
        engine.state.public.last_bid = Bid(5, 4)
        view = engine.get_view(0)
        agent = RandomAgent(rng=None)
        # ask agent to choose action many times, ensure no bid > total dice occurs
        for _ in range(50):
            act = agent.choose_action(view)
            if isinstance(act, BidAction):
                q = act.bid.quantity
                self.assertLessEqual(q, 6)


if __name__ == '__main__':
    unittest.main()

