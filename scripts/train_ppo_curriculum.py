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
    League-based self-play training phase.
    
    The agent trains against a diverse pool of opponents including:
    - All curriculum agents (to prevent catastrophic forgetting)
    - Frozen versions of itself from previous iterations
    
    This "league" approach helps maintain performance against diverse strategies
    while improving through self-play.
    
    Args:
        base_model_path: Path to the base model to start from
        timesteps: Timesteps per self-play iteration
        num_iterations: Number of self-play iterations
        enable_early_stopping: If True, stops when win_rate_threshold is reached
        win_rate_threshold: Win rate threshold for early stopping (default: 0.95)
    """
    print("\n" + "="*80)
    print("LEAGUE-BASED SELF-PLAY TRAINING")
    print("="*80 + "\n")
    
    game_config = GameConfig(
        num_players=2,
        total_dice=5,
        faces=(1, 2, 3, 4, 5, 6),
        ones_wild=False
    )
    
    # Build league of curriculum opponents (sample from these to prevent forgetting)
    curriculum = get_curriculum_stages()
    league_opponents = []
    
    print("Building opponent league from curriculum agents...")
    for opponent_cls, name, _ in curriculum:
        try:
            # Test if agent can be instantiated
            test_agent = opponent_cls()
            league_opponents.append((opponent_cls, name))
            print(f"  ✓ Added {name} to league")
        except UntrainedAgentException:
            print(f"  ⚠ Skipped {name} (not trained)")
        except Exception as e:
            print(f"  ⚠ Skipped {name} ({e})")
    
    print(f"\nLeague size: {len(league_opponents)} curriculum agents\n")
    
    current_path = base_model_path
    frozen_models = []  # Track frozen self versions
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*80}")
        print(f"SELF-PLAY ITERATION {iteration}/{num_iterations}")
        print(f"{'='*80}\n")
        
        # Create a COPY of the frozen model so it doesn't get overwritten during training
        frozen_model_path = f"{current_path}_frozen_iter{iteration}"
        shutil.copy2(f"{current_path}.zip", f"{frozen_model_path}.zip")
        print(f"Created frozen copy: {frozen_model_path}.zip")
        
        # Create a frozen stochastic opponent from current model
        class FrozenSelfAgent(PPOAgent):
            _frozen_path = frozen_model_path  # Class variable to capture path
            
            def __init__(self):
                super().__init__(model_path=self._frozen_path)
            
            def choose_action(self, view):
                """Override to use stochastic policy for better self-play diversity."""
                if self.model is None:
                    raise RuntimeError("PPOAgent has no loaded model.")

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
                
                # KEY: Use stochastic play for diversity
                action_idx, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
                
                game_action = self._decode_action(action_idx)
                self._record_action(is_me=True, action=game_action)
                self.last_bid_on_table = public.last_bid
                
                return game_action
        
        FrozenSelfAgent.__name__ = f"FrozenSelf_v{iteration}"
        frozen_models.append((FrozenSelfAgent, f"FrozenSelf_v{iteration}"))
        
        # Build complete league: curriculum + all frozen selves
        complete_league = league_opponents + frozen_models
        print(f"Training against league of {len(complete_league)} opponents:")
        print(f"  - {len(league_opponents)} curriculum agents")
        print(f"  - {len(frozen_models)} frozen self versions\n")
        
        # Create a league opponent that randomly samples from the pool
        import random as py_random
        
        class LeagueOpponent:
            """Opponent that randomly samples from a league of agents each episode."""
            def __init__(self):
                self.league = complete_league
                self.current_agent = None
                self.current_name = None
                self._select_opponent()
            
            def _select_opponent(self):
                """Randomly select an opponent from the league."""
                opponent_cls, name = py_random.choice(self.league)
                self.current_agent = opponent_cls()
                self.current_name = name
            
            def choose_action(self, view):
                """Delegate to current opponent."""
                return self.current_agent.choose_action(view)
            
            def reset_for_new_episode(self):
                """Called at start of each episode to resample opponent."""
                self._select_opponent()
        
        LeagueOpponent.__name__ = f"League_{len(complete_league)}agents"
        
        # Train against the league, always save to ppo_model.zip
        saved_path = train_ppo_agent(
            opponent_cls=LeagueOpponent,
            game_config=game_config,
            load_path=current_path,
            save_name="ppo_model",  # Always overwrite main model
            total_timesteps=timesteps,
            log_interval=10,
            enable_early_stopping=enable_early_stopping,
            win_rate_threshold=win_rate_threshold
        )
        
        # Clean up the frozen copy (we keep track of it via the class definition)
        # Note: We don't delete yet because LeagueOpponent might still reference it
        
        # Update for next iteration
        current_path = saved_path
        print(f"\nSelf-play iteration {iteration} completed. Model updated at: {saved_path}.zip")
    
    # Cleanup all frozen copies after training completes
    print(f"\nCleaning up temporary frozen model copies...")
    for i in range(1, num_iterations + 1):
        frozen_path = f"{base_model_path}_frozen_iter{i}.zip"
        if os.path.exists(frozen_path):
            os.remove(frozen_path)
            print(f"  Removed: {frozen_path}")
    
    print(f"\nLeague-based self-play training complete. Final model: {current_path}.zip")


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
