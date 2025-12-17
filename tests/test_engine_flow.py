import unittest
from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine
from liars_dice.core.bid import Bid
from liars_dice.core.actions import BidAction, CallLiarAction


class TestEngineFlow(unittest.TestCase):
    def test_full_round_call_liar(self):
        cfg = GameConfig(rng_seed=1)
        engine = GameEngine(cfg)
        engine.start_new_round()
        # player0 makes a valid bid
        engine.apply_action(0, BidAction(Bid(1, 2)))
        # player1 calls liar
        engine.apply_action(1, CallLiarAction())
        self.assertTrue(engine.is_terminal())
        ev = engine.get_events()
        # expect RoundEnded event present
        self.assertTrue(any(e.get('type') == 'RoundEnded' for e in ev))


if __name__ == '__main__':
    unittest.main()

