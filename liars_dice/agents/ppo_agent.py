import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from . import register_agent
from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.core.bid import Bid
from liars_dice.core.actions import CallLiarAction, BidAction
from liars_dice.agents.reinforcement_agent.encoder import HistoryObservationEncoder
from liars_dice.agents.reinforcement_agent.config import MODEL_CONFIG, TRAINING_CONFIG
from liars_dice.agents.reinforcement_agent.env import LiarsDiceGymEnv 


@register_agent("rl_ppo")
class PPOAgent(Agent):
    """
    Inference Wrapper for the PPO Model.
    Inherits from Agent to be compatible with GameEngine and Tournament scripts.
    This agent uses a HistoryObservationEncoder to encode observations, and loads it's weights from agents/weights/ppo_model.zip.
    """
    def __init__(self, model_path=None):
        # Default to the path in config if none provided
        if model_path is None:
            model_path = TRAINING_CONFIG["model_save_path"]
            
        self.encoder = None # Will init lazily or via setup
        self.model = None
        self.history_buffer = []
        self.last_round_idx = -1
        self.last_bid_on_table = None
        
        # Load Model
        full_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        if os.path.exists(full_path):
            self.model = MaskablePPO.load(full_path)
        else:
            raise UntrainedAgentException(
                f"PPO model not found at {full_path}. Train the model first using scripts/train_ppo_curriculum.py"
            )

        # Init Encoder params
        # Note: We don't have game config yet (choose_action receives view later)
        # We assume standard 5 dice for now or set it on first call
        self.total_dice = 5 
        self.encoder = HistoryObservationEncoder(total_dice=self.total_dice, history_len=TRAINING_CONFIG["history_length"])

    def choose_action(self, view):
        if self.model is None:
            raise RuntimeError("PPOAgent has no loaded model.")

        # 1. Sync History (Detect Opponent Moves)
        self._sync_history(view)

        # 2. Prepare Data for Encoder
        # View is a dict with keys: player_id (optional), public, my_dice, config
        my_dice = view["my_dice"]  # tuple of dice values
        my_hand = {}
        for die in my_dice:
            my_hand[die] = my_hand.get(die, 0) + 1
        
        my_dice_count = len(my_dice)
        
        # Get opponent dice count from public state
        public = view["public"]
        player_id = view.get("player_id", 0)  # Default to 0 if not provided
        opp_dice_count = sum(c for i, c in enumerate(public.dice_counts) if i != player_id)
        
        # 3. Encode - Use the history buffer maintained by this agent
        # Temporarily swap encoder's history buffer with ours
        old_buffer = self.encoder.history_buffer
        self.encoder.history_buffer = self.history_buffer
        obs = self.encoder.encode(my_hand, my_dice_count, opp_dice_count)
        self.encoder.history_buffer = old_buffer

        # 4. Mask
        mask = self._get_action_mask(view)

        # 5. Predict
        action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        
        # 6. Decode
        game_action = self._decode_action(action_idx)

        # 7. Record My Own Action
        self._record_action(is_me=True, action=game_action)
        public = view["public"]
        self.last_bid_on_table = public.last_bid 
        
        return game_action

    def _sync_history(self, view):
        # A. New Round Reset
        public = view["public"]
        if public.round_index != self.last_round_idx:
            self.history_buffer = []
            self.last_round_idx = public.round_index
            self.last_bid_on_table = None
            return

        # B. Detect Opponent Move
        current_bid = public.last_bid
        if current_bid != self.last_bid_on_table:
            if current_bid is not None:
                # Opponent placed a bid
                opp_action = BidAction(Bid(quantity=current_bid.quantity, face=current_bid.face))
                self._record_action(is_me=False, action=opp_action)
            else:
                # Bid is None but last was not None -> Round reset (Handled by A) or error
                pass
        self.last_bid_on_table = current_bid

    def _record_action(self, is_me, action_type=None, bid_obj=None, action=None):
        # Support both action object and separate parameters
        if action is not None:
            action_type = type(action).__name__  # "BidAction" or "CallLiarAction"
            bid_obj = action.bid if action_type == "BidAction" else None
        
        # Encode action into history vector
        is_bid = 1.0 if action_type == "BidAction" else 0.0
        qty = bid_obj.quantity / self.encoder.max_bid_qty if bid_obj else 0.0
        face = bid_obj.face / 6.0 if bid_obj else 0.0
        is_me_val = 1.0 if is_me else 0.0
        
        h_vec = [is_me_val, is_bid, qty, face]
        self.history_buffer.append(h_vec)

    def _get_action_mask(self, view):
        # Must match Gym Env logic EXACTLY
        max_qty = self.encoder.max_bid_qty # derived from total dice
        n_actions = 1 + (max_qty * 6)
        mask = np.zeros(n_actions, dtype=bool)
        
        public = view["public"]
        curr = public.last_bid
        
        if curr is not None:
            mask[0] = True # Call Liar
            
        for q in range(1, max_qty + 1):
            for f in range(1, 7):
                idx = 1 + (q-1)*6 + (f-1)
                if idx < n_actions:
                    cand = Bid(q, f)
                    if curr is None or cand.is_higher_than(curr):
                        mask[idx] = True
        return mask

    def _decode_action(self, idx):
        if idx == 0: return CallLiarAction()
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
                   total_timesteps=None, log_interval=10, enable_early_stopping=True, 
                   win_rate_threshold=0.95):
    """
    Trains the PPO Agent against a specific opponent class.
    Supports curriculum learning by allowing continued training from a checkpoint.
    
    Args:
        opponent_cls: The class of the opponent (e.g. RandomAgent).
        game_config: The GameConfig object.
        load_path: Optional path to load an existing model to continue training.
        save_name: Name for the saved model file.
        total_timesteps: Number of timesteps to train. If None, uses config default.
        log_interval: How often (in episodes) to print training progress.
        enable_early_stopping: If True, stops when win_rate_threshold is reached.
        win_rate_threshold: Win rate threshold for early stopping (default: 0.95 = 95%).
    
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
        model.tensorboard_log = TRAINING_CONFIG["log_dir"]
    else:
        print("Initializing new PPO model...", flush=True)
        print(f"Training for {total_timesteps} timesteps\n", flush=True)
        # Use opponent name for clean TensorBoard organization
        tb_log_name = opponent_name
        model = MaskablePPO(
            MODEL_CONFIG["policy_type"],
            env,
            verbose=0,
            tensorboard_log=TRAINING_CONFIG["log_dir"],
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
    
    print(f"\n{'='*60}", flush=True)
    print(f"Training finished. Model saved to {save_path}.zip", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    env.close()
    return save_path