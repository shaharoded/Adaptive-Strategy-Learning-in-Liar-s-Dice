"""
train_ppo_curriculum.py

Curriculum training script for PPO agent.
Trains the PPO agent progressively against increasingly difficult opponents.

Usage:
    python scripts/train_ppo_curriculum.py [--resume] [--timesteps 100000]
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine
from liars_dice.agents.base import UntrainedAgentException
from liars_dice.agents.ppo_agent import train_ppo_agent, PPOAgent
from liars_dice.agents import AGENT_MAP
from liars_dice.agents.ppo_agent import PPOAgent


def evaluate_agent_against_all(agent_path, game_config, num_games=100, additional_opponents=None):
    """
    Evaluate the agent against all potential opponents without training.
    
    Args:
        agent_path: Path to the agent model
        game_config: Game configuration
        num_games: Number of games per opponent
        additional_opponents: Optional list of (opponent_class, name) tuples to evaluate (e.g., frozen selves)
    
    Returns:
        dict: {opponent_name: (wins, total_games, win_rate)}
    """    
    # Load the agent
    try:
        agent = PPOAgent(model_path=agent_path)
    except Exception as e:
        print(f"Failed to load agent: {e}")
        return {}
    
    # Get all potential opponents from AGENT_MAP
    all_opponents = get_league_opponents()
    
    # Add any additional opponents (e.g., frozen selves)
    if additional_opponents:
        all_opponents.extend(additional_opponents)
    
    results = {}
    
    for opp_cls, opp_name in all_opponents:
        try:
            wins = 0
            for game_num in range(num_games):
                opponent = opp_cls()
                
                # Alternate who goes first
                if game_num % 2 == 0:
                    agents = [agent, opponent]
                    agent_player = 0
                else:
                    agents = [opponent, agent]
                    agent_player = 1
                
                # Run match with dice elimination
                engine = GameEngine(game_config)
                dice_counts = [game_config.total_dice, game_config.total_dice]
                
                while min(dice_counts) > 0:
                    # Update dice counts BEFORE starting round so dice are rolled correctly
                    for i in range(2):
                        engine.state.players[i].num_dice = dice_counts[i]
                    engine.start_new_round()
                    
                    # Play round
                    while not engine.is_terminal():
                        current_player = engine.state.public.current_player
                        view = engine.get_view(current_player)
                        action = agents[current_player].choose_action(view)
                        
                        try:
                            engine.apply_action(current_player, action)
                        except Exception:
                            # Error = other player wins
                            winner = 1 - current_player
                            break
                    
                    if engine.is_terminal():
                        round_loser = engine.state.public.loser
                        dice_counts[round_loser] -= 1
                
                # Match winner
                winner = 0 if dice_counts[0] > 0 else 1
                if winner == agent_player:
                    wins += 1
            
            win_rate = wins / num_games
            results[opp_name] = (wins, num_games, win_rate)
                
        except UntrainedAgentException:
            print(f"SKIPPED (not trained)")
        except Exception as e:
            print(f"ERROR: {e}")
    
    return results


def print_evaluation_table(results, threshold=0.95):
    """
    Print a formatted table of evaluation results.
    
    Args:
        results: dict from evaluate_agent_against_all
        threshold: Win rate threshold to highlight weaknesses
    """   
    print(f"{'Opponent':<30} {'Win Rate':<15} {'Wins/Games'}")
    print(f"{'-'*30} {'-'*15} {'-'*15}")
    
    for opp_name in sorted(results.keys()):
        wins, games, win_rate = results[opp_name]
        bar_length = int(win_rate * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        # Highlight opponents below threshold
        marker = "⚠️ " if win_rate < threshold else "✓ "
        print(f"{marker}{opp_name:<28} {win_rate:>6.1%} {bar}  {wins}/{games}")
    
    # Summary stats
    if results:
        avg_win_rate = sum(r[2] for r in results.values()) / len(results)
        weak_opponents = [name for name, (_, _, wr) in results.items() if wr < threshold]
        
        print(f"\n{'='*80}")
        print(f"Average win rate: {avg_win_rate:.1%}")
        print(f"Opponents below {threshold:.0%} threshold: {len(weak_opponents)}")
        
        if weak_opponents:
            print(f"\nWeaknesses identified: {', '.join(weak_opponents)}")
            print(f"Next training passes will focus on these {len(weak_opponents)} opponent(s)")
        else:
            print(f"\n🎉 All opponents mastered! Agent beats all opponents at {threshold:.0%}+")
        
        print(f"{'='*80}\n")


def get_league_opponents():
    """
    Get all registered opponents from AGENT_MAP for the league.
    
    Excludes:
    - rl_ppo (the agent being trained)
    - Any agents that fail to instantiate (e.g., untrained agents)
    
    Returns list of (opponent_class, display_name) tuples.
    """
    elite_opponents = []
    
    for agent_name, agent_cls in AGENT_MAP.items():
        # Skip the PPO agent itself
        if agent_name == "rl_ppo":
            continue
        
        # Try to instantiate to verify it's available
        try:
            # Use the class name as display name
            display_name = agent_cls.__name__
            elite_opponents.append((agent_cls, display_name))
        except UntrainedAgentException:
            # Skip agents that aren't trained yet (e.g., Nash/CFR might not be trained)
            pass
        except Exception:
            # Skip agents that fail to instantiate for other reasons
            pass
    
    return elite_opponents


def get_curriculum_stages():
    """
    Define the curriculum stages using registered agents from AGENT_MAP.
    
    Streamlined curriculum focusing on essential opponents:
    1. random - Learn basic game mechanics
    2. probability_min_raise - Learn conservative probability-based play
    3. bayesian - Learn against probability-based reasoning
    4. nash_cfr - Learn game-theoretically optimal play (if available)
    
    Returns list of (opponent_class, display_name, timesteps) tuples.
    """
    curriculum = []
    
    # Define desired curriculum with agent registry names
    desired_stages = [
        ("random", 100_000),
        ("probability_min_raise", 100_000),
        ("bayesian", 150_000),
        ("nash_cfr", 200_000),
    ]
    
    for agent_name, timesteps in desired_stages:
        if agent_name in AGENT_MAP:
            agent_cls = AGENT_MAP[agent_name]
            display_name = agent_cls.__name__
            curriculum.append((agent_cls, display_name, timesteps))
    
    return curriculum


def train_curriculum(resume=False, base_timesteps=None, stages=None, enable_early_stopping=True,
                    win_rate_threshold=0.95, run_dynamic_league=True, **league_kwargs):
    """
    Execute curriculum training, followed by the extended loss-weighted phase.
    
    Args:
        resume: If True, attempts to resume from the last saved checkpoint
        base_timesteps: Override default timesteps per stage (optional)
        stages: List of specific stage indices to train (optional, trains all if None)
        enable_early_stopping: If True, stops each stage when win_rate_threshold is reached
        win_rate_threshold: Win rate threshold for early stopping (default: 0.95 = 95%)
        run_dynamic_league: If True, automatically runs the extended loss-weighted phase after curriculum
        **league_kwargs: Additional arguments to pass to extended_curriculum_training
    
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
    
    # Handle fresh start vs resume
    model_exists = os.path.exists(base_path + ".zip")
    
    if not resume and model_exists:
        # Fresh start requested - delete existing model
        print(f"🗑️  Fresh start requested - deleting existing model: {base_path}.zip")
        os.remove(base_path + ".zip")
        current_model_path = None
        print(f"   Starting training from scratch\n")
    elif resume and model_exists:
        # Resume from existing model
        current_model_path = base_path
        print(f"ℹ️  Resuming training from existing model: {base_path}.zip\n")
    elif model_exists:
        # Model exists but no explicit choice - auto-resume
        current_model_path = base_path
        print(f"ℹ️  Found existing model - will continue training from: {base_path}.zip")
        print(f"   (Use --fresh-start to start from scratch)\n")
    else:
        # No model exists - start fresh
        current_model_path = None
        print(f"ℹ️  No existing model found - starting fresh\n")
    
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
    print("FIRST STEP CURRICULUM TRAINING COMPLETE!")
    print("="*80 + "\n")
    
    # Automatically run extended curriculum training
    if run_dynamic_league:
        print("Starting curriculum training phase (loss-weighted)...\n")
        extended_curriculum_training(
            base_model_path=base_path,
            enable_early_stopping=enable_early_stopping,
            win_rate_threshold=win_rate_threshold,
            **league_kwargs
        )
    else:
        print("To visualize training progress, run:")
        print("  tensorboard --logdir=./runs/ppo_training/")
        print("\n")


def extended_curriculum_training(base_model_path, timesteps=500_000, 
                                  enable_early_stopping=True, win_rate_threshold=0.95,
                                  evaluation_games=50, eval_interval_steps=50_000):
    """
    Extended curriculum training using loss-weighted opponent sampling.
    
    Strategy:
    1. Train against ALL base opponents simultaneously
    2. Sample opponents proportional to loss rate (train more against weak opponents)
    3. Evaluate periodically against fixed opponent set (no dynamic league)
    4. Continue until win rates stabilize (true convergence)
    
    Args:
        base_model_path: Path to the base model to start from
        timesteps: Total training timesteps
        enable_early_stopping: If True, stops when avg win rate plateaus
        win_rate_threshold: Target average win rate
        evaluation_games: Number of games per opponent during evaluation
        eval_interval_steps: Steps between evaluation checks
    """
    print("\n" + "="*80)
    print("EXTENDED CURRICULUM TRAINING (Loss-Weighted)")
    print(f"Total timesteps: {timesteps:,} | Eval interval: {eval_interval_steps:,}")
    print("="*80 + "\n")
    
    game_config = GameConfig(
        num_players=2,
        total_dice=5,
        faces=(1, 2, 3, 4, 5, 6),
        ones_wild=False
    )
    
    # Get all base opponents (fixed throughout training)
    elite_opponents = get_league_opponents()
    
    if not elite_opponents:
        print("⚠️  No opponents available for extended curriculum training")
        return
    
    print(f"Training against {len(elite_opponents)} base opponents:")
    for cls, name in elite_opponents:
        print(f"  - {name}")
    print()
    
    # Helper to build a weighted opponent-sampling class with no-arg constructor
    def make_weighted_opponent(opponents, weights, display_name):
        norm_w = np.array(weights, dtype=float)
        if norm_w.sum() <= 0:
            norm_w = np.ones(len(opponents), dtype=float)
        norm_w = norm_w / norm_w.sum()

        class WeightedOpponent:
            def __init__(self):
                self.league = opponents
                self.weights = norm_w
                self.current_agent = None
                self.current_name = None
                self._select_opponent()

            def _select_opponent(self):
                idx = np.random.choice(len(self.league), p=self.weights)
                opponent_cls, name = self.league[idx]
                self.current_agent = opponent_cls()
                self.current_name = name

            def choose_action(self, view):
                return self.current_agent.choose_action(view)

            def reset_for_new_episode(self):
                self._select_opponent()

        WeightedOpponent.__name__ = display_name
        return WeightedOpponent
    
    # Initial training with uniform weights
    print("Phase 1: Initial training with uniform opponent sampling...")
    current_path = base_model_path
    best_avg_wr = 0.0
    eval_results_history = []
    
    # Train for initial phase
    UniformOpponent = make_weighted_opponent(
        opponents=elite_opponents,
        weights=[1.0] * len(elite_opponents),
        display_name=f"AllOpponents_{len(elite_opponents)}"
    )

    saved_path = train_ppo_agent(
        opponent_cls=UniformOpponent,
        game_config=game_config,
        load_path=current_path,
        save_name="ppo_model",
        total_timesteps=timesteps,
        log_interval=10,
        enable_early_stopping=False,  # Manual eval-based stopping
        win_rate_threshold=0.99  # Very high to avoid early stop
    )
    current_path = saved_path
    
    # Phase 2: Periodic evaluation with loss-weighted resampling
    print("\n" + "="*80)
    print("Phase 2: Loss-weighted sampling with periodic evaluation")
    print("="*80 + "\n")
    
    steps_since_eval = 0
    eval_count = 0
    evals_without_improvement = 0
    max_evals_without_improvement = 5  # Stop if 5 evals in a row show no improvement
    
    while evals_without_improvement < max_evals_without_improvement:
        eval_count += 1
        
        print(f"\n{'='*80}")
        print(f"EVALUATION CHECKPOINT {eval_count}")
        print(f"{'='*80}\n")
        
        # Evaluate current model
        eval_results = evaluate_agent_against_all(
            agent_path=current_path,
            game_config=game_config,
            num_games=evaluation_games,
            additional_opponents=None  # No frozen selves - fixed evaluation set
        )
        
        print_evaluation_table(eval_results, threshold=0.95)
        
        # Calculate metrics
        if eval_results:
            avg_wr = sum(r[2] for r in eval_results.values()) / len(eval_results)
        else:
            avg_wr = 0.0
        
        eval_results_history.append((eval_count, avg_wr))
        
        print(f"\n📊 Average win rate: {avg_wr:.1%}")
        print(f"📈 Best average win rate: {best_avg_wr:.1%}")
        
        # Check for improvement
        if avg_wr > best_avg_wr:
            print(f"✅ IMPROVED by {(avg_wr - best_avg_wr)*100:.1f}%!")
            best_avg_wr = avg_wr
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1
            print(f"❌ No improvement ({evals_without_improvement}/{max_evals_without_improvement})")
        
        # Check for convergence
        if avg_wr >= win_rate_threshold:
            print(f"\n🎉 TARGET REACHED! Average win rate {avg_wr:.1%} >= {win_rate_threshold:.0%}")
            print(f"Training complete after {eval_count} evaluation checkpoints")
            break
        
        if evals_without_improvement >= max_evals_without_improvement:
            print(f"\n⚠️  No improvement for {max_evals_without_improvement} evaluations")
            print(f"Stopping training. Final win rate: {best_avg_wr:.1%}")
            break
        
        # Update sampling weights for next training phase
        print(f"\nUpdating opponent sampling weights based on current loss rates...")
        loss_rates = {name: 1.0 - wr for name, (_, _, wr) in eval_results.items()}
        weights = [loss_rates.get(name, 0.5) for (_, name) in elite_opponents]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0 / len(elite_opponents)] * len(elite_opponents)
        else:
            weights = [w / total_w for w in weights]

        print("New sampling distribution:")
        for (_, name), weight in zip(elite_opponents, weights):
            loss_rate = loss_rates.get(name, 0.5)
            print(f"  {name:<25} Loss: {loss_rate:.1%} Weight: {weight:.1%}")

        # Build a new opponent sampler class with updated weights
        WeightedOpponent = make_weighted_opponent(
            opponents=elite_opponents,
            weights=weights,
            display_name=f"AllOpponents_{len(elite_opponents)}_LW"
        )

        # Train for next phase
        print(f"\nPhase {eval_count + 2}: Training for {eval_interval_steps:,} steps...")
        saved_path = train_ppo_agent(
            opponent_cls=WeightedOpponent,
            game_config=game_config,
            load_path=current_path,
            save_name="ppo_model",
            total_timesteps=eval_interval_steps,
            log_interval=10,
            enable_early_stopping=False,
            win_rate_threshold=0.99
        )
        current_path = saved_path
    
    # Final summary
    print(f"\n{'='*80}")
    print("EXTENDED CURRICULUM TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Final average win rate: {best_avg_wr:.1%}")
    print(f"Evaluation checkpoints: {eval_count}")
    print(f"Final model: {current_path}.zip")
    
    # Plot convergence if history exists
    if eval_results_history:
        print(f"\nConvergence history:")
        for checkpoint, wr in eval_results_history:
            bar_len = int(wr * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  Checkpoint {checkpoint:2d}: {wr:.1%} {bar}")



def main():
    parser = argparse.ArgumentParser(description="Curriculum training for PPO agent with extended loss-weighted curriculum")
    parser.add_argument("--resume", action="store_true", 
                       help="Explicitly resume from existing checkpoint (deprecated - auto-resumes by default)")
    parser.add_argument("--fresh-start", action="store_true",
                       help="Ignore existing model and start training from scratch")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Override timesteps per stage")
    parser.add_argument("--stages", type=int, nargs="+", default=None,
                       help="Train only specific stages (0-indexed)")
    parser.add_argument("--disable-early-stopping", action="store_true",
                       help="Disable early stopping based on win rate (applies to curriculum)")
    parser.add_argument("--win-rate-threshold", type=float, default=0.95,
                       help="Win rate threshold for training (default: 0.95)")
    
    # Extended curriculum arguments
    parser.add_argument("--skip-extended", action="store_true",
                       help="Skip extended curriculum (only run initial curriculum)")
    parser.add_argument("--extended-timesteps", type=int, default=500_000,
                       help="Total timesteps for extended curriculum phase (default: 500k)")
    parser.add_argument("--eval-interval", type=int, default=50_000,
                       help="Steps between evaluation checkpoints (default: 50k)")
    parser.add_argument("--evaluation-games", type=int, default=50,
                       help="Number of games per opponent during evaluation (default: 50)")
    
    args = parser.parse_args()
    
    # Run curriculum training with extended curriculum
    train_curriculum(
        resume=not args.fresh_start,  # Auto-resume by default
        base_timesteps=args.timesteps,
        stages=args.stages,
        enable_early_stopping=not args.disable_early_stopping,
        win_rate_threshold=args.win_rate_threshold,
        run_dynamic_league=not args.skip_extended,
        # Extended curriculum kwargs
        timesteps=args.extended_timesteps,
        eval_interval_steps=args.eval_interval,
        evaluation_games=args.evaluation_games
    )


if __name__ == "__main__":
    main()
