"""
test_ppo_setup.py

Unit tests to verify PPO agent setup is working correctly.
Tests encoder, environment, and training initialization.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from liars_dice.core.config import GameConfig

# Try importing PPO dependencies
try:
    from liars_dice.agents.reinforcement_agent.encoder import HistoryObservationEncoder
    from liars_dice.agents.reinforcement_agent.env import LiarsDiceGymEnv
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from liars_dice.agents.reinforcement_agent.config import MODEL_CONFIG
    PPO_AVAILABLE = True
except ImportError as e:
    PPO_AVAILABLE = False
    IMPORT_ERROR = str(e)

from liars_dice.agents.random_agent import RandomAgent
from liars_dice.core.bid import Bid
from liars_dice.core.actions import BidAction, CallLiarAction


@unittest.skipUnless(PPO_AVAILABLE, f"PPO dependencies not available: {IMPORT_ERROR if not PPO_AVAILABLE else ''}")
class TestPPOEncoder(unittest.TestCase):
    """Test cases for HistoryObservationEncoder."""
    
    def test_encoder_initialization_with_config(self):
        """Test encoder initialization with game_config."""
        game_config = GameConfig(num_players=2, total_dice=5)
        encoder = HistoryObservationEncoder(game_config=game_config, history_len=10)
        
        self.assertEqual(encoder.max_dice, 5)
        self.assertEqual(encoder.history_len, 10)
        self.assertEqual(encoder.obs_dim, 6 + 3 + 40)  # hand + context + history
    
    def test_encoder_initialization_with_total_dice(self):
        """Test encoder initialization with total_dice parameter."""
        encoder = HistoryObservationEncoder(total_dice=10, history_len=10)
        
        self.assertEqual(encoder.max_dice, 10)
        self.assertEqual(encoder.history_len, 10)
    
    def test_encoder_encoding(self):
        """Test observation encoding."""
        encoder = HistoryObservationEncoder(total_dice=10, history_len=10)
        hand_dict = {1: 1, 3: 2, 5: 1, 6: 1}
        my_dice_count = 5
        opp_dice_count = 5
        
        obs = encoder.encode(hand_dict, my_dice_count, opp_dice_count)
        
        self.assertEqual(obs.shape, encoder.shape)
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0), "Observation values should be normalized [0, 1]")
    
    def test_encoder_add_event(self):
        """Test adding events to history buffer."""
        encoder = HistoryObservationEncoder(total_dice=10, history_len=10)
        encoder.reset()
        
        action1 = BidAction(Bid(2, 3))
        encoder.add_event(player_id=0, action=action1, agent_id=0)
        
        self.assertEqual(len(encoder.history_buffer), 1)
        
        action2 = BidAction(Bid(3, 3))
        encoder.add_event(player_id=1, action=action2, agent_id=0)
        
        self.assertEqual(len(encoder.history_buffer), 2)
        # First value should indicate if it's "my" action
        self.assertEqual(encoder.history_buffer[0][0], 1.0)  # Player 0's action (me)
        self.assertEqual(encoder.history_buffer[1][0], 0.0)  # Player 1's action (not me)
        # Second value should indicate if it's a bid (1.0) or call liar (0.0)
        self.assertEqual(encoder.history_buffer[0][1], 1.0)  # Both are bids
        self.assertEqual(encoder.history_buffer[1][1], 1.0)


@unittest.skipUnless(PPO_AVAILABLE, "PPO dependencies not available")
class TestPPOEnvironment(unittest.TestCase):
    """Test cases for LiarsDiceGymEnv."""
    
    def setUp(self):
        """Set up test environment."""
        self.game_config = GameConfig(num_players=2, total_dice=5)
        self.env = LiarsDiceGymEnv(self.game_config, RandomAgent)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_environment_spaces(self):
        """Test action and observation spaces are correctly defined."""
        self.assertIsNotNone(self.env.action_space)
        self.assertIsNotNone(self.env.observation_space)
        
        # Check action space size (1 for Call Liar + N for bids)
        expected_actions = 1 + (self.env.max_bid_quantity * 6)
        self.assertEqual(self.env.action_space.n, expected_actions)
    
    def test_environment_reset(self):
        """Test environment reset returns valid observation and info."""
        obs, info = self.env.reset()
        
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIn('action_mask', info)
        self.assertTrue(info['action_mask'].sum() > 0, "Should have at least one valid action")
    
    def test_environment_step(self):
        """Test environment step with valid action."""
        obs, info = self.env.reset()
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask)[0]
        
        self.assertGreater(len(valid_actions), 0, "Should have valid actions")
        
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)


@unittest.skipUnless(PPO_AVAILABLE, "PPO dependencies not available")
class TestPPOTrainingSetup(unittest.TestCase):
    """Test cases for PPO training setup."""
    
    def test_model_initialization(self):
        """Test that MaskablePPO model can be created."""
        game_config = GameConfig(num_players=2, total_dice=5)
        env = Monitor(LiarsDiceGymEnv(game_config, RandomAgent))
        
        try:
            model = MaskablePPO(
                MODEL_CONFIG["policy_type"],
                env,
                verbose=0,
                learning_rate=MODEL_CONFIG["learning_rate"],
                gamma=MODEL_CONFIG["gamma"],
                batch_size=MODEL_CONFIG["batch_size"],
                ent_coef=MODEL_CONFIG["ent_coef"],
                policy_kwargs=MODEL_CONFIG["policy_kwargs"]
            )
            
            self.assertIsNotNone(model)
            
            # Test prediction
            obs, _ = env.reset()
            action_mask = env.get_wrapper_attr('get_action_mask')()
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=False)
            
            self.assertTrue(action_mask[action], "Predicted action should be valid according to mask")
        
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
