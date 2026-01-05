"""
Smoke tests for all registered agents.
Tests that each agent can be instantiated, called without errors, and returns valid actions.
This is a technical "are you working" test, not a strategy quality test.
"""

import unittest
from liars_dice.agents import AGENT_MAP
from liars_dice.core.config import GameConfig
from liars_dice.core.bid import Bid
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.agents.base import UntrainedAgentException


class TestAgentSmoke(unittest.TestCase):
    """Smoke tests to verify all agents can be instantiated and produce valid actions."""

    def setUp(self):
        """Set up test fixtures with common game configurations and views."""
        self.config = GameConfig(dice_distribution=(5, 5), rng_seed=42)
        self.faces = (1, 2, 3, 4, 5, 6)
        
        # Sample dice for testing
        self.my_dice = [3, 3, 4, 5, 6]
        
        # Create a PublicStateView helper similar to engine's get_view
        def make_public_view(last_bid, turn_index, current_player, dice_counts):
            """Create a public view with dice_counts like the engine does."""
            class PublicView:
                def __init__(self):
                    self.dice_counts = dice_counts
                    self.last_bid = last_bid
                    self.turn_index = turn_index
                    self.current_player = current_player
                    self.bid_history = []
                    self.status = "BIDDING"
                    self.winner = None
                    self.loser = None
                    self.round_index = 0
            return PublicView()
        
        # View with no last bid (opening state)
        self.view_no_bid = {
            "public": make_public_view(None, 0, 0, [5, 5]),
            "my_dice": self.my_dice,
            "config": self.config
        }
        
        # View with a low bid
        self.view_low_bid = {
            "public": make_public_view(Bid(2, 3), 3, 1, [5, 5]),
            "my_dice": self.my_dice,
            "config": self.config
        }
        
        # View with a high bid
        self.view_high_bid = {
            "public": make_public_view(Bid(8, 6), 10, 0, [5, 5]),
            "my_dice": self.my_dice,
            "config": self.config
        }
        
        # View with impossible bid (more than total dice)
        self.view_impossible_bid = {
            "public": make_public_view(Bid(15, 4), 5, 1, [5, 5]),
            "my_dice": self.my_dice,
            "config": self.config
        }
        
        # View with few dice remaining
        self.view_low_dice = {
            "public": make_public_view(Bid(1, 2), 2, 0, [2, 1]),
            "my_dice": [3, 4],
            "config": GameConfig(dice_distribution=(2, 1), rng_seed=42)
        }

    def test_all_agents_registered(self):
        """Verify that AGENT_MAP contains at least some agents."""
        self.assertGreater(len(AGENT_MAP), 0, "AGENT_MAP should have registered agents")
        print(f"\nFound {len(AGENT_MAP)} registered agents: {list(AGENT_MAP.keys())}")

    def test_agents_can_instantiate(self):
        """Test that all registered agents can be instantiated without errors."""
        failed_agents = []
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
                self.assertIsNotNone(agent, f"Agent {name} instantiation returned None")
            except UntrainedAgentException as e:
                # Nash CFR agent may not have trained weights, skip
                print(f"  Skipping {name}: {e}")
            except Exception as e:
                failed_agents.append((name, str(e)))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Failed to instantiate agents:\n{error_msg}")

    def test_agents_handle_no_bid(self):
        """Test that all agents can handle the opening state (no last bid)."""
        failed_agents = []
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
            except UntrainedAgentException:
                continue
            except Exception as e:
                failed_agents.append((name, f"Instantiation: {e}"))
                continue
            
            try:
                action = agent.choose_action(self.view_no_bid)
                self.assertIsNotNone(action, f"Agent {name} returned None action")
                self.assertIsInstance(action, (BidAction, CallLiarAction), 
                                    f"Agent {name} returned invalid action type: {type(action)}")
                
                # For opening bid, should always be BidAction
                if isinstance(action, CallLiarAction):
                    print(f"  Warning: {name} called liar on opening (no last bid)")
                
            except Exception as e:
                failed_agents.append((name, f"No bid state: {e}"))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Agents failed to handle no-bid state:\n{error_msg}")

    def test_agents_handle_low_bid(self):
        """Test that all agents can handle a normal low bid scenario."""
        failed_agents = []
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
            except UntrainedAgentException:
                continue
            except Exception as e:
                failed_agents.append((name, f"Instantiation: {e}"))
                continue
            
            try:
                action = agent.choose_action(self.view_low_bid)
                self.assertIsNotNone(action, f"Agent {name} returned None action")
                self.assertIsInstance(action, (BidAction, CallLiarAction), 
                                    f"Agent {name} returned invalid action type: {type(action)}")
                
                # If it's a bid, verify it's higher than last bid
                if isinstance(action, BidAction):
                    last_bid = self.view_low_bid["public"].last_bid
                    self.assertTrue(action.bid.is_higher_than(last_bid), 
                                  f"Agent {name} made invalid bid: {action.bid} not higher than {last_bid}")
                
            except Exception as e:
                failed_agents.append((name, f"Low bid state: {e}"))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Agents failed to handle low-bid state:\n{error_msg}")

    def test_agents_handle_high_bid(self):
        """Test that all agents can handle a high bid scenario."""
        failed_agents = []
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
            except UntrainedAgentException:
                continue
            except Exception as e:
                failed_agents.append((name, f"Instantiation: {e}"))
                continue
            
            try:
                action = agent.choose_action(self.view_high_bid)
                self.assertIsNotNone(action, f"Agent {name} returned None action")
                self.assertIsInstance(action, (BidAction, CallLiarAction), 
                                    f"Agent {name} returned invalid action type: {type(action)}")
                
                # If it's a bid, verify it's valid
                if isinstance(action, BidAction):
                    last_bid = self.view_high_bid["public"].last_bid
                    total_dice = sum(self.view_high_bid["public"].dice_counts)
                    self.assertTrue(action.bid.is_higher_than(last_bid), 
                                  f"Agent {name} made invalid bid: {action.bid} not higher than {last_bid}")
                    self.assertLessEqual(action.bid.quantity, total_dice,
                                       f"Agent {name} bid more dice than exist: {action.bid.quantity} > {total_dice}")
                
            except Exception as e:
                failed_agents.append((name, f"High bid state: {e}"))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Agents failed to handle high-bid state:\n{error_msg}")

    def test_agents_handle_impossible_bid(self):
        """Test that all agents properly handle impossible bids (should call liar or raise appropriately)."""
        failed_agents = []
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
            except UntrainedAgentException:
                continue
            except Exception as e:
                failed_agents.append((name, f"Instantiation: {e}"))
                continue
            
            try:
                action = agent.choose_action(self.view_impossible_bid)
                self.assertIsNotNone(action, f"Agent {name} returned None action")
                self.assertIsInstance(action, (BidAction, CallLiarAction), 
                                    f"Agent {name} returned invalid action type: {type(action)}")
                
                # For impossible bids (15 dice when only 10 exist), should ideally call liar
                # But we just check it doesn't crash
                if isinstance(action, BidAction):
                    print(f"  Note: {name} raised instead of calling liar on impossible bid (15 dice, 10 total)")
                
            except Exception as e:
                failed_agents.append((name, f"Impossible bid state: {e}"))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Agents failed to handle impossible-bid state:\n{error_msg}")

    def test_agents_handle_low_dice_count(self):
        """Test that all agents can handle endgame scenarios with few dice."""
        failed_agents = []
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
            except UntrainedAgentException:
                continue
            except Exception as e:
                failed_agents.append((name, f"Instantiation: {e}"))
                continue
            
            try:
                action = agent.choose_action(self.view_low_dice)
                self.assertIsNotNone(action, f"Agent {name} returned None action")
                self.assertIsInstance(action, (BidAction, CallLiarAction), 
                                    f"Agent {name} returned invalid action type: {type(action)}")
                
                # If it's a bid, verify it's valid for low dice count
                if isinstance(action, BidAction):
                    total_dice = sum(self.view_low_dice["public"].dice_counts)
                    self.assertLessEqual(action.bid.quantity, total_dice,
                                       f"Agent {name} bid more dice than exist: {action.bid.quantity} > {total_dice}")
                
            except Exception as e:
                failed_agents.append((name, f"Low dice count state: {e}"))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Agents failed to handle low-dice-count state:\n{error_msg}")

    def test_agents_return_valid_actions_consistently(self):
        """Test that agents return valid actions consistently across multiple calls."""
        failed_agents = []
        num_trials = 5
        
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
            except UntrainedAgentException:
                continue
            except Exception as e:
                failed_agents.append((name, f"Instantiation: {e}"))
                continue
            
            try:
                # Test multiple calls with same view
                for i in range(num_trials):
                    action = agent.choose_action(self.view_low_bid)
                    self.assertIsNotNone(action, f"Agent {name} returned None on trial {i+1}")
                    self.assertIsInstance(action, (BidAction, CallLiarAction), 
                                        f"Agent {name} returned invalid type on trial {i+1}: {type(action)}")
                    
                    # Validate bid actions
                    if isinstance(action, BidAction):
                        last_bid = self.view_low_bid["public"].last_bid
                        self.assertTrue(action.bid.is_higher_than(last_bid),
                                      f"Agent {name} made invalid bid on trial {i+1}")
                
            except Exception as e:
                failed_agents.append((name, f"Consistency test: {e}"))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Agents failed consistency test:\n{error_msg}")

    def test_agents_have_required_method(self):
        """Test that all agents have the required choose_action method."""
        failed_agents = []
        for name, agent_class in AGENT_MAP.items():
            try:
                agent = agent_class()
            except UntrainedAgentException:
                continue
            except Exception as e:
                failed_agents.append((name, f"Instantiation: {e}"))
                continue
            
            if not hasattr(agent, 'choose_action'):
                failed_agents.append((name, "Missing choose_action method"))
            elif not callable(getattr(agent, 'choose_action')):
                failed_agents.append((name, "choose_action is not callable"))
        
        if failed_agents:
            error_msg = "\n".join([f"  {name}: {err}" for name, err in failed_agents])
            self.fail(f"Agents missing required methods:\n{error_msg}")


if __name__ == "__main__":
    unittest.main()
