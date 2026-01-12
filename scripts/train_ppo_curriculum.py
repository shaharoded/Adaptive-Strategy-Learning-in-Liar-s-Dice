"""
train_ppo_curriculum.py

Curriculum training script for PPO agent.
Trains the PPO agent progressively against increasingly difficult opponents.

Usage:
    python scripts/train_ppo_curriculum.py [--resume] [--timesteps 100000]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from liars_dice.core.config import GameConfig
from liars_dice.agents.base import UntrainedAgentException
from liars_dice.agents.ppo_agent import train_ppo_agent, PPOAgent
from liars_dice.agents.random_agent import RandomAgent, CautiousRandomAgent, AggressiveRandomAgent
from liars_dice.agents.heuristic_agent import (
    ConservativeAgent,
    AggressiveAgent,
    ProbabilityMinRaiseAgent,
    ProbabilityMaxRaiseAgent,
    MinRaiseAgent,
    MaxRaiseAgent,
    MirrorAgent,
    MaxCountBidAgent,
    RandomFaceAgent,
    SafeFaceAgent,
    BluffingAgent,
    ThresholdLiarAgent,
    ChaoticSafeAgent,
    ChaoticUnsafeAgent,
    AlternatorAgent,
    CycleFaceAgent,
    ParityAgent,
    RandomThresholdAgent
)
from liars_dice.agents.bayesian_agent import BayesianAgent
from liars_dice.agents.nash_agent import NashCFRAgent


def get_curriculum_stages():
    """
    Define the curriculum stages: (opponent_class, display_name, timesteps)
    
    The curriculum is designed to progressively increase difficulty:
    1. Random agents (baseline - learn basic game mechanics)
    2. Simple heuristic agents (rule-based strategies)
    3. Advanced heuristic agents (more sophisticated strategies)
    4. Bayesian agent (probability-based reasoning)
    5. Nash/CFR agent (game-theoretically optimal play)
    6. Self-play (frozen versions of itself)
    """
    return [
        # Stage 1: Learn basic game mechanics against random agents
        (RandomAgent, "Random", 30_000),
        (CautiousRandomAgent, "CautiousRandom", 30_000),
        (AggressiveRandomAgent, "AggressiveRandom", 30_000),
        
        # Stage 2: Simple heuristic strategies
        (MinRaiseAgent, "MinRaise", 40_000),
        (MaxRaiseAgent, "MaxRaise", 40_000),
        (ConservativeAgent, "Conservative", 40_000),
        (AggressiveAgent, "Aggressive", 40_000),
        
        # Stage 3: More complex heuristics
        (ProbabilityMinRaiseAgent, "ProbabilityMinRaise", 50_000),
        (ProbabilityMaxRaiseAgent, "ProbabilityMaxRaise", 50_000),
        (MirrorAgent, "Mirror", 50_000),
        (MaxCountBidAgent, "MaxCount", 50_000),
        (RandomFaceAgent, "RandomFace", 50_000),
        (SafeFaceAgent, "SafeFace", 50_000),
        
        # Stage 4: Specialized and tricky heuristics
        (BluffingAgent, "Bluffing", 60_000),
        (ThresholdLiarAgent, "ThresholdLiar", 60_000),
        (AlternatorAgent, "Alternator", 60_000),
        (CycleFaceAgent, "CycleFace", 60_000),
        (ParityAgent, "Parity", 60_000),
        (RandomThresholdAgent, "RandomThreshold", 60_000),
        (ChaoticSafeAgent, "ChaoticSafe", 60_000),
        (ChaoticUnsafeAgent, "ChaoticUnsafe", 60_000),
        
        # Stage 5: Advanced agents
        (BayesianAgent, "Bayesian", 100_000),
        
        # Stage 6: Nash/CFR agent (most difficult, game-theoretically sound)
        # Note: This will only work if NashCFRAgent has been trained
        (NashCFRAgent, "NashCFR", 150_000),
        
        # Stage 7: Self-play is handled separately below
    ]


def train_curriculum(resume=False, base_timesteps=None, stages=None, enable_early_stopping=True,
                    win_rate_threshold=0.95):
    """
    Execute curriculum training.
    
    Args:
        resume: If True, attempts to resume from the last saved checkpoint
        base_timesteps: Override default timesteps per stage (optional)
        stages: List of specific stage indices to train (optional, trains all if None)
        enable_early_stopping: If True, stops each stage when win_rate_threshold is reached
        win_rate_threshold: Win rate threshold for early stopping (default: 0.95 = 95%)
    
    Note: Agents that aren't trained (Nash/CFR) are automatically skipped if UntrainedAgentException is raised.
    """
    print("\n" + "="*80)
    print("PPO CURRICULUM TRAINING")
    if enable_early_stopping:
        print(f"Early stopping enabled: Each stage will stop at {win_rate_threshold:.0%} win rate")
    print("="*80 + "\n")
    
    # Setup game configuration
    game_config = GameConfig(
        num_players=2,
        total_dice=5,  # total_dice is per-player count
        faces=(1, 2, 3, 4, 5, 6),
        ones_wild=False
    )
    
    curriculum = get_curriculum_stages()
    
    if stages is not None:
        curriculum = [curriculum[i] for i in stages if i < len(curriculum)]
    
    # Path for the main model (gets updated after each stage)
    base_path = "./liars_dice/agents/weights/ppo_model"
    
    # Auto-resume if model exists (unless explicitly starting fresh)
    model_exists = os.path.exists(base_path + ".zip")
    current_model_path = base_path if (resume or model_exists) else None
    
    if model_exists and not resume:
        print(f"ℹ️  Found existing model - will continue training from: {base_path}.zip")
        print(f"   (Use --fresh-start to ignore existing model)\n")
    
    print(f"Training Configuration:")
    print(f"  Game: {game_config.num_players} players, {game_config.total_dice} dice each")
    print(f"  Curriculum Stages: {len(curriculum)}")
    print(f"  Resume Training: {resume}")
    print(f"  Base Model Path: {base_path}")
    print("\n")
    
    # Train through each curriculum stage
    for stage_idx, (opponent_cls, opponent_name, timesteps) in enumerate(curriculum, 1):
        if base_timesteps is not None:
            timesteps = base_timesteps
            
        print(f"\n{'='*80}")
        print(f"CURRICULUM STAGE {stage_idx}/{len(curriculum)}: {opponent_name}")
        print(f"{'='*80}\n")
        
        # Handle potential errors (e.g., untrained Nash agent)
        try:
            # Train against this opponent (always save to same ppo_model.zip)
            saved_path = train_ppo_agent(
                opponent_cls=opponent_cls,
                game_config=game_config,
                load_path=current_model_path,  # Load from previous stage or None
                save_name="ppo_model",  # Always save to same file
                total_timesteps=timesteps,
                log_interval=10,
                enable_early_stopping=enable_early_stopping,
                win_rate_threshold=win_rate_threshold
            )
            
            # Update the current model path for the next stage
            current_model_path = saved_path
            
            print(f"\nStage {stage_idx} completed. Model saved to: {saved_path}.zip")
            print(f"Continuing to next stage...\n")
            
        except Exception as e:
            if isinstance(e, UntrainedAgentException):
                print(f"\n⚠️  Skipping Stage {stage_idx} ({opponent_name}): Agent not yet trained")
                print(f"   (Train {opponent_name} first or it will be skipped automatically)\n")
            else:
                print(f"\n⚠️  Warning: Stage {stage_idx} ({opponent_name}) failed with error: {e}")
                print(f"Skipping this stage and continuing with next...\n")
            # Continue with the same model path
            continue
    
    # Curriculum complete
    print("\n" + "="*80)
    print("CURRICULUM TRAINING COMPLETE!")
    print(f"Final model: {base_path}.zip")
    print("="*80 + "\n")
    
    print("To visualize training progress, run:")
    print("  tensorboard --logdir=./runs/ppo_training/")
    print("\n")


def self_play_training(base_model_path, timesteps=100_000, num_iterations=3,
                       enable_early_stopping=True, win_rate_threshold=0.95):
    """
    Additional self-play training phase.
    
    The agent trains against frozen versions of itself, which helps it discover
    and exploit weaknesses in its own strategy.
    
    Args:
        base_model_path: Path to the base model to start from
        timesteps: Timesteps per self-play iteration
        num_iterations: Number of self-play iterations
        enable_early_stopping: If True, stops when win_rate_threshold is reached
        win_rate_threshold: Win rate threshold for early stopping (default: 0.95)
    """
    print("\n" + "="*80)
    print("SELF-PLAY TRAINING")
    print("="*80 + "\n")
    
    game_config = GameConfig(
        num_players=2,
        total_dice=5,
        faces=(1, 2, 3, 4, 5, 6),
        ones_wild=False
    )
    
    current_path = base_model_path
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*80}")
        print(f"SELF-PLAY ITERATION {iteration}/{num_iterations}")
        print(f"{'='*80}\n")
        
        # Create a COPY of the frozen model so it doesn't get overwritten during training
        frozen_model_path = f"{current_path}_frozen_iter{iteration}"
        shutil.copy2(f"{current_path}.zip", f"{frozen_model_path}.zip")
        print(f"Created frozen copy: {frozen_model_path}.zip")
        
        # Create a frozen opponent class that uses the copied frozen model
        # IMPORTANT: Use stochastic play (deterministic=False) to avoid first-player advantage
        class FrozenSelfAgent(PPOAgent):
            def __init__(self):
                super().__init__(model_path=frozen_model_path)
            
            def choose_action(self, view):
                """Override to use stochastic policy for better self-play diversity."""
                if self.model is None:
                    raise RuntimeError("PPOAgent has no loaded model.")

                # Sync history and prepare observation (same as parent)
                self._sync_history(view)
                
                my_dice = view["my_dice"]
                my_hand = {}
                for die in my_dice:
                    my_hand[die] = my_hand.get(die, 0) + 1
                
                my_dice_count = len(my_dice)
                public = view["public"]
                player_id = view.get("player_id", 0)
                opp_dice_count = sum(c for i, c in enumerate(public.dice_counts) if i != player_id)
                
                old_buffer = self.encoder.history_buffer
                self.encoder.history_buffer = self.history_buffer
                obs = self.encoder.encode(my_hand, my_dice_count, opp_dice_count)
                self.encoder.history_buffer = old_buffer
                
                mask = self._get_action_mask(view)
                
                # KEY CHANGE: Use deterministic=False for stochastic self-play
                action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
                
                game_action = self._decode_action(action_idx)
                self._record_action(is_me=True, action=game_action)
                self.last_bid_on_table = public.last_bid
                
                return game_action
        
        FrozenSelfAgent.__name__ = f"FrozenSelf_v{iteration}"
        
        # Train against the frozen version, always save to ppo_model.zip
        saved_path = train_ppo_agent(
            opponent_cls=FrozenSelfAgent,
            game_config=game_config,
            load_path=current_path,
            save_name="ppo_model",  # Always overwrite main model
            total_timesteps=timesteps,
            log_interval=10,
            enable_early_stopping=enable_early_stopping,
            win_rate_threshold=win_rate_threshold
        )
        
        # Clean up the frozen copy
        if os.path.exists(f"{frozen_model_path}.zip"):
            os.remove(f"{frozen_model_path}.zip")
            print(f"Removed temporary frozen copy: {frozen_model_path}.zip")
        
        # Update for next iteration
        current_path = saved_path
        print(f"\nSelf-play iteration {iteration} completed. Model updated at: {saved_path}.zip")
    
    print(f"\nSelf-play training complete. Final model: {current_path}.zip")


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for PPO agent")
    parser.add_argument("--resume", action="store_true", 
                       help="Explicitly resume from existing checkpoint (deprecated - auto-resumes by default)")
    parser.add_argument("--fresh-start", action="store_true",
                       help="Ignore existing model and start training from scratch")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Override timesteps per stage")
    parser.add_argument("--stages", type=int, nargs="+", default=None,
                       help="Train only specific stages (0-indexed)")
    parser.add_argument("--self-play", action="store_true",
                       help="Run self-play training after curriculum")
    parser.add_argument("--self-play-iterations", type=int, default=3,
                       help="Number of self-play iterations")
    parser.add_argument("--disable-early-stopping", action="store_true",
                       help="Disable early stopping based on win rate (applies to both curriculum and self-play)")
    parser.add_argument("--win-rate-threshold", type=float, default=0.95,
                       help="Win rate threshold for early stopping (default: 0.95, applies to both curriculum and self-play)")
    
    args = parser.parse_args()
    
    # Run curriculum training (auto-resume unless fresh-start)
    train_curriculum(
        resume=not args.fresh_start,  # Auto-resume by default
        base_timesteps=args.timesteps,
        stages=args.stages,
        enable_early_stopping=not args.disable_early_stopping,
        win_rate_threshold=args.win_rate_threshold
    )
    
    # Optionally run self-play
    if args.self_play:
        base_path = "./liars_dice/agents/weights/ppo_model"
        self_play_training(
            base_model_path=base_path,
            timesteps=args.timesteps or 100_000,
            num_iterations=args.self_play_iterations,
            enable_early_stopping=not args.disable_early_stopping,
            win_rate_threshold=args.win_rate_threshold
        )


if __name__ == "__main__":
    main()
