import unittest
from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine
from liars_dice.core.bid import Bid
from liars_dice.core.actions import BidAction, CallLiarAction


class TestTurnLog(unittest.TestCase):
    """
    Tests for the per-turn `turn_log` snapshots recorded by `GameEngine`.
    These tests verify:
      - An initial snapshot is recorded at the start of the round.
      - After each action a snapshot is appended containing actor, action, public and players info.
      - Final snapshot reflects round end when a liar call resolves.
    """

    def test_initial_and_action_snapshots(self):
        cfg = GameConfig()
        engine = GameEngine(cfg)
        engine.start_new_round()
        # initial snapshot should be present
        self.assertGreaterEqual(len(engine.turn_log), 1)
        initial = engine.turn_log[0]
        self.assertIsNone(initial['actor'])
        self.assertIsNone(initial['action'])
        # perform a bid and check snapshot appended
        engine.apply_action(0, BidAction(Bid(1, 2)))
        self.assertGreaterEqual(len(engine.turn_log), 2)
        second = engine.turn_log[-1]
        self.assertEqual(second['actor'], 0)
        self.assertEqual(second['action']['type'], 'Bid')
        self.assertIn('public', second)
        self.assertIn('players', second)

    def test_final_snapshot_on_call(self):
        cfg = GameConfig()
        engine = GameEngine(cfg)
        engine.start_new_round()
        engine.apply_action(0, BidAction(Bid(1, 2)))
        engine.apply_action(1, CallLiarAction())
        # after call the public status should be ENDED
        self.assertEqual(engine.state.public.status, 'ENDED')
        # the most recent snapshot should include the action actor and reflect ended state
        last = engine.turn_log[-1]
        self.assertEqual(last['action']['type'], 'CallLiar')
        self.assertEqual(last['public']['status'], 'ENDED')


if __name__ == '__main__':
    unittest.main()

