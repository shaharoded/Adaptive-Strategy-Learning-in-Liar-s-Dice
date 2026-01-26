"""
PPO Agent for Liar's Dice

Proximal Policy Optimization agent using MaskablePPO from sb3-contrib.
Uses history-based observation encoding and action masking for valid bids.
Supports both training and inference modes.
"""

import os
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from . import register_agent
from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.agents.reinforcement_agent_utils.encoder import HistoryObservationEncoder
from liars_dice.agents.reinforcement_agent_utils.config import MODEL_CONFIG, TRAINING_CONFIG
from liars_dice.agents.reinforcement_agent_utils.env import LiarsDiceGymEnv
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid 


@register_agent("rl_ppo")
class PPOAgent(Agent):
    """
    PPO agent for Liar's Dice using history-based observations.
    
    Supports two modes:
    - Auto-sync (disable_auto_sync=False): Automatically tracks opponent actions
    - Manual-sync (disable_auto_sync=True): Relies on external history management (for adaptive agent)
    """
    
    def __init__(self, model_path=None, stochastic=False, disable_auto_sync=False):
        """
        Args:
            model_path: Path to trained model (.zip). Uses default from config if None.
            stochastic: Use stochastic policy (training) vs deterministic (evaluation).
            disable_auto_sync: Disable auto history tracking (for use with adaptive agent).
        """
        # Resolve model path
        if model_path is None:
            model_path = TRAINING_CONFIG["model_save_path"]
        
        model_path = Path(model_path)
        if model_path.suffix != ".zip":
            model_path = model_path.with_suffix(".zip")
        if not model_path.is_absolute():
            model_path = (Path(__file__).parent / model_path).resolve()
        
        # Load model
        full_path = str(model_path)
        if os.path.exists(full_path):
            self.model = MaskablePPO.load(full_path)
        else:
            raise UntrainedAgentException(
                f"PPO model not found at {full_path}. Train using scripts/train_ppo_curriculum.py"
            )
        
        # Initialize state
        self.encoder = HistoryObservationEncoder(total_dice=5, history_len=MODEL_CONFIG["history_length"])
        self.history_buffer = []
        self.last_round_idx = -1
        self.last_bid_on_table = None
        self.stochastic = stochastic
        self.disable_auto_sync = disable_auto_sync
    
    def choose_action(self, view):
        """Select action using trained PPO policy."""
        if self.model is None:
            raise RuntimeError("PPOAgent has no loaded model.")

        # Auto-sync history if enabled
        if not self.disable_auto_sync:
            self._sync_history(view)
        
        # Prepare observation
        my_dice = view["my_dice"]
        my_hand = {die: list(my_dice).count(die) for die in set(my_dice)}
        my_dice_count = len(my_dice)
        
        public = view["public"]
        player_id = view.get("player_id", 0)
        opp_dice_count = sum(c for i, c in enumerate(public.dice_counts) if i != player_id)
        
        self.encoder.history_buffer = self.history_buffer
        obs = self.encoder.encode(my_hand, my_dice_count, opp_dice_count)
        
        # Predict action
        mask = self._get_action_mask(view)
        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=not self.stochastic)
        game_action = self._decode_action(action_idx)
        
        # Record action if auto-sync enabled
        if not self.disable_auto_sync:
            self._record_action(is_me=True, action=game_action)
            self.last_bid_on_table = public.last_bid
        
        return game_action
    
    def set_last_bid(self, bid):
        """Manually set bid tracker (used by adaptive agent)."""
        self.last_bid_on_table = bid

    def reset(self):
        """Reset agent state for new game."""
        self.history_buffer = []
        self.last_round_idx = -1
        self.last_bid_on_table = None

    def _sync_history(self, view):
        """Auto-detect and record opponent actions from state changes."""
        public = view["public"]
        
        # Reset on new round
        if public.round_index != self.last_round_idx:
            self.history_buffer = []
            self.last_round_idx = public.round_index
            self.last_bid_on_table = None
            return

        # Detect opponent bid
        current_bid = public.last_bid
        if current_bid != self.last_bid_on_table and current_bid is not None:
            opp_action = BidAction(Bid(quantity=current_bid.quantity, face=current_bid.face))
            self._record_action(is_me=False, action=opp_action)
        
        self.last_bid_on_table = current_bid

    def _record_action(self, is_me, action_type=None, bid_obj=None, action=None):
        """Encode action into history vector."""
        if action is not None:
            action_type = type(action).__name__
            bid_obj = action.bid if action_type == "BidAction" else None
        
        is_bid = 1.0 if action_type == "BidAction" else 0.0
        qty = bid_obj.quantity / self.encoder.max_bid_qty if bid_obj else 0.0
        face = bid_obj.face / 6.0 if bid_obj else 0.0
        is_me_val = 1.0 if is_me else 0.0
        
        self.history_buffer.append([is_me_val, is_bid, qty, face])

    def _get_action_mask(self, view):
        """Generate mask for valid actions."""
        max_qty = self.encoder.max_bid_qty
        n_actions = 1 + (max_qty * 6)
        mask = np.zeros(n_actions, dtype=bool)
        
        public = view["public"]
        curr = public.last_bid
        
        if curr is not None:
            mask[0] = True  # Call Liar always valid if bid exists
            
        for q in range(1, max_qty + 1):
            for f in range(1, 7):
                idx = 1 + (q-1)*6 + (f-1)
                if idx < n_actions:
                    cand = Bid(q, f)
                    if curr is None or cand.is_higher_than(curr):
                        mask[idx] = True
        return mask

    def _decode_action(self, idx):
        """Convert action index to game action."""
        if idx == 0:
            return CallLiarAction()
        adj = int(idx) - 1
        return BidAction(Bid((adj // 6) + 1, (adj % 6) + 1))


# --- Custom Callback for TensorBoard Logging ---

class RewardLoggingCallback(BaseCallback):
    """
    Custom callback to log episode rewards and other metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones")[0]:
            # Log episode reward
            info = self.locals.get("infos")[0]
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Log to TensorBoard
                self.logger.record("rollout/ep_reward", episode_reward)
                self.logger.record("rollout/ep_length", episode_length)
                
                # Calculate and log moving averages (last 100 episodes)
                if len(self.episode_rewards) >= 100:
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    avg_length = np.mean(self.episode_lengths[-100:])
                    self.logger.record("rollout/ep_reward_mean_100", avg_reward)
                    self.logger.record("rollout/ep_length_mean_100", avg_length)
        
        return True


class WinRateCallback(BaseCallback):
    """
    Custom callback to track win rate and stop training when threshold is reached.
    
    Tracks wins over the last N episodes and stops training when win rate exceeds threshold.
    """
    def __init__(self, win_rate_threshold=0.95, window_size=100, verbose=1, print_freq=100):
        super().__init__(verbose)
        self.win_rate_threshold = win_rate_threshold
        self.window_size = window_size
        self.print_freq = print_freq  # Print every N episodes
        self.episode_outcomes = []  # Store 1 for win, 0 for loss
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            # Check if match_winner is in info (indicates match completion with dice elimination)
            if "match_winner" in info:
                # match_winner == 0 means RL agent won
                did_win = 1 if info["match_winner"] == 0 else 0
                self.episode_outcomes.append(did_win)
                self.total_episodes += 1
                
                # Keep only the last window_size episodes
                if len(self.episode_outcomes) > self.window_size:
                    self.episode_outcomes.pop(0)
                
                # Calculate win rate over the window
                if len(self.episode_outcomes) >= self.window_size:
                    win_rate = sum(self.episode_outcomes) / len(self.episode_outcomes)
                    
                    # Log to TensorBoard (every episode)
                    self.logger.record("performance/win_rate_100", win_rate)
                    self.logger.record("performance/total_matches", self.total_episodes)
                    
                    # Print to console only every print_freq episodes
                    if self.verbose >= 1 and self.total_episodes % self.print_freq == 0:
                        print(f"[Win Rate] Matches: {self.total_episodes:,} | "
                              f"Win Rate (last {self.window_size}): {win_rate:.1%}")
                    
                    # Check if threshold is reached
                    if win_rate >= self.win_rate_threshold:
                        if self.verbose >= 1:
                            print(f"\\n{'='*60}")
                            print(f"🎯 TARGET ACHIEVED! Win rate {win_rate:.1%} >= {self.win_rate_threshold:.0%}")
                            print(f"Stopping training after {self.total_episodes:,} matches.")
                            print(f"{'='*60}\\n")
                        return False  # Stop training
        
        return True  # Continue training


# --- Training Function ---

def train_ppo_agent(opponent_cls, game_config, load_path=None, save_name="ppo_model", 
                   total_timesteps=None, enable_early_stopping=True, 
                   win_rate_threshold=0.95, log_dir=None):
    """
    Trains the PPO Agent against a specific opponent class.
    Supports curriculum learning by allowing continued training from a checkpoint.
    
    Args:
        opponent_cls: The class of the opponent (e.g. RandomAgent).
        game_config: The GameConfig object.
        load_path: Optional path to load an existing model to continue training.
        save_name: Name for the saved model file.
        total_timesteps: Number of timesteps to train. If None, uses config default.
        enable_early_stopping: If True, stops when win_rate_threshold is reached.
        win_rate_threshold: Win rate threshold for early stopping (default: 0.95 = 95%).
        log_dir: Optional TensorBoard log directory. If None, uses config default.
    
    Returns:
        str: Path to the saved model.
    """
    opponent_name = opponent_cls.__name__
    print(f"\n{'='*60}", flush=True)
    print(f"Starting Training vs {opponent_name}", flush=True)
    if enable_early_stopping:
        print(f"Early stopping enabled: Will stop at {win_rate_threshold:.0%} win rate (last 100 matches)", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    if total_timesteps is None:
        total_timesteps = TRAINING_CONFIG["total_timesteps"]
    
    if log_dir is None:
        log_dir = TRAINING_CONFIG["log_dir"]
    
    # 1. Setup Environment with ActionMasker wrapper
    base_env = LiarsDiceGymEnv(game_config, opponent_cls)
    
    # ActionMasker wrapper requires a mask_fn that takes env and returns the mask
    def mask_fn(env):
        return env.get_action_mask()
    
    env = ActionMasker(base_env, mask_fn)
    env = Monitor(env)

    # 2. Initialize or Load Model
    model_exists = load_path and os.path.exists(load_path + ".zip")
    
    if model_exists:
        print(f"Loading existing model from {load_path}.zip...", flush=True)
        print(f"Continuing training for {total_timesteps} more timesteps\n", flush=True)
        model = MaskablePPO.load(load_path, env=env)
        # Use consistent opponent name for TensorBoard organization
        tb_log_name = opponent_name
        model.tensorboard_log = log_dir
    else:
        print("Initializing new PPO model...", flush=True)
        print(f"Training for {total_timesteps} timesteps\n", flush=True)
        # Use opponent name for clean TensorBoard organization
        tb_log_name = opponent_name
        model = MaskablePPO(
            MODEL_CONFIG["policy_type"],
            env,
            verbose=0,
            tensorboard_log=log_dir,
            learning_rate=MODEL_CONFIG["learning_rate"],
            gamma=MODEL_CONFIG["gamma"],
            batch_size=MODEL_CONFIG["batch_size"],
            ent_coef=MODEL_CONFIG["ent_coef"],
            policy_kwargs=MODEL_CONFIG["policy_kwargs"]
        )

    # 3. Setup Callbacks
    callbacks = []
    
    # Add win rate callback if early stopping is enabled
    if enable_early_stopping:
        win_rate_callback = WinRateCallback(
            win_rate_threshold=win_rate_threshold,
            window_size=100,
            verbose=1,
            print_freq=100  # Print every 100 matches
        )
        callbacks.append(win_rate_callback)

    # 4. Train
    print(f"Training against {opponent_name}...", flush=True)
    print(f"Progress updates every 100 matches. Full stats in TensorBoard.\n", flush=True)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name,
        reset_num_timesteps=False if model_exists else True,
        progress_bar=True
    )

    # 5. Save
    save_dir = os.path.dirname(TRAINING_CONFIG["model_save_path"])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    model.save(save_path)
    
    env.close()
    return save_path