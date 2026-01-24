"""
train_adaptive_agent.py

Train the adaptive agent system (specialist experts + neural classifier).

Usage:
    # Train all experts + neural classifier
    python scripts/train_adaptive_agent.py --train-all
    
    # Train specific expert for adaptive agent
    python scripts/train_adaptive_agent.py --opponent random --timesteps 200000
    
    # Train only the neural classifier
    python scripts/train_adaptive_agent.py --train-classifier --samples 1000
    
    # Evaluate all trained experts
    python scripts/train_adaptive_agent.py --evaluate-experts
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from liars_dice.core.config import GameConfig
from liars_dice.agents.random_agent import RandomAgent, CautiousRandomAgent, AggressiveRandomAgent
from liars_dice.agents.heuristic_agent import (
    ConservativeAgent, 
    AggressiveAgent,
    ProbabilityMinRaiseAgent,
    ProbabilityMaxRaiseAgent,
    MinRaiseAgent,
    MaxRaiseAgent,
    MirrorAgent,
    MaxCountBidAgent
)
from liars_dice.agents.bayesian_agent import BayesianAgent
from liars_dice.agents import AGENT_MAP
from liars_dice.agents.adapter_agent.adaptive_training import (
    train_specialist_expert,
    train_neural_classifier,
    evaluate_specialist_experts
)
from liars_dice.agents.adapter_agent.config import EXPERT_CONFIG, CLASSIFIER_CONFIG, PATH_CONFIG


# Available opponents
OPPONENTS = {
    "random": RandomAgent,
    "random_cautious": CautiousRandomAgent,
    "random_aggressive": AggressiveRandomAgent,
    "conservative": ConservativeAgent,
    "aggressive": AggressiveAgent,
    "probability_min": ProbabilityMinRaiseAgent,
    "probability_max": ProbabilityMaxRaiseAgent,
    "minraise": MinRaiseAgent,
    "maxraise": MaxRaiseAgent,
    "mirror": MirrorAgent,
    "maxcount": MaxCountBidAgent,
    "bayesian": BayesianAgent,
}


def train_adaptive_system(
    game_config: GameConfig,
    opponent_classes: dict,
    base_model_path: Optional[str] = None,
    timesteps_per_expert: Optional[int] = None,
    classifier_samples: Optional[int] = None,
    win_rate_threshold: Optional[float] = None
):
    """
    Train the complete adaptive agent system:
    1. Train specialist experts for each opponent
    2. Train the neural LSTM classifier
    3. Evaluate all experts
    
    Args:
        game_config: Game configuration
        opponent_classes: Dict mapping opponent names to classes
        base_model_path: Path to generalist PPO for warm start (default: from PATH_CONFIG)
        timesteps_per_expert: Training timesteps per specialist (default: from EXPERT_CONFIG)
        classifier_samples: Number of games per opponent for classifier (default: 1000)
        win_rate_threshold: Target win rate for experts (default: from EXPERT_CONFIG)
    """
    # Load defaults from config
    if base_model_path is None:
        base_model_path = PATH_CONFIG["generalist_model"]
    if timesteps_per_expert is None:
        timesteps_per_expert = EXPERT_CONFIG["timesteps"]
    if win_rate_threshold is None:
        win_rate_threshold = EXPERT_CONFIG["win_rate_threshold"]
    if classifier_samples is None:
        classifier_samples = 1000
    
    print("\n" + "="*80)
    print("ADAPTIVE AGENT SYSTEM TRAINING (Neural LSTM)")
    print("="*80)
    print(f"\nPhase 1: Training {len(opponent_classes)} Specialist Experts")
    print(f"  Target win rate: {win_rate_threshold:.1%}")
    print(f"  Timesteps per expert: {timesteps_per_expert:,}")
    print(f"  Warm start from: {base_model_path}")
    print("="*80 + "\n")
    
    # Phase 1: Train all specialist experts
    trained_experts = []
    for opp_name, opp_cls in opponent_classes.items():
        try:
            expert_path = train_specialist_expert(
                opponent_cls=opp_cls,
                opponent_name=opp_name,
                game_config=game_config,
                base_model_path=base_model_path if os.path.exists(base_model_path + ".zip") else None,
                total_timesteps=timesteps_per_expert,
                win_rate_threshold=win_rate_threshold
            )
            trained_experts.append((opp_name, expert_path))
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to train expert for {opp_name}: {e}")
            print("Continuing with remaining opponents...\n")
            continue
    
    if not trained_experts:
        print("\n❌ No experts were successfully trained. Cannot continue.")
        return
    
    print(f"\n✓ Phase 1 Complete: {len(trained_experts)}/{len(opponent_classes)} experts trained\n")
    
    # Phase 2: Train neural classifier
    print("\n" + "="*80)
    print(f"Phase 2: Training Neural LSTM Classifier")
    print(f"  Trajectories per opponent: {classifier_samples}")
    print(f"  Architecture: {CLASSIFIER_CONFIG['num_lstm_layers']}-layer LSTM")
    print(f"  Hidden dim: {CLASSIFIER_CONFIG['hidden_dim']}")
    print("="*80 + "\n")
    
    try:
        classifier = train_neural_classifier(
            opponent_classes=opponent_classes,
            game_config=game_config,
            samples_per_opponent=classifier_samples,
            save_path=None,  # Uses PATH_CONFIG default
            device="cpu"  # Can be changed to "cuda" if GPU available
        )
        print("✓ Phase 2 Complete: Neural classifier trained\n")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to train classifier: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Phase 3: Evaluate all experts
    print("\n" + "="*80)
    print("Phase 3: Evaluating Specialist Experts")
    print("="*80 + "\n")
    
    try:
        results = evaluate_specialist_experts(
            opponent_classes=opponent_classes,
            game_config=game_config,
            num_games=100
        )
        
        # Summary
        successful_experts = sum(1 for _, _, wr in results.values() if wr >= win_rate_threshold)
        print(f"\n{'='*80}")
        print(f"ADAPTIVE SYSTEM TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"\nExperts meeting {win_rate_threshold:.0%} threshold: {successful_experts}/{len(results)}")
        print(f"Models saved in: {PATH_CONFIG['adaptive_models_dir']}")
        print(f"Expert logs: {EXPERT_CONFIG['log_dir']}")
        print(f"Classifier logs: {PATH_CONFIG['tensorboard_log_dir']}")
        print(f"\nTo use the adaptive agent:")
        print(f"  from liars_dice.agents.adaptive_agent import AdaptiveAgent")
        print(f"  agent = AdaptiveAgent()")
        print()
        
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to evaluate experts: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Train adaptive agent system")
    
    # Mode selection
    parser.add_argument("--train-all", action="store_true",
                       help="Train all experts + neural classifier")
    parser.add_argument("--train-classifier", action="store_true",
                       help="Train only the neural classifier")
    parser.add_argument("--evaluate-experts", action="store_true",
                       help="Evaluate all trained experts")
    
    # Opponent selection (for training single expert)
    parser.add_argument("--opponent", type=str, choices=list(OPPONENTS.keys()),
                       help="Opponent agent type (for single expert training)")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=None,
                       help=f"Number of training timesteps (default: {EXPERT_CONFIG['timesteps']:,})")
    
    # Game configuration
    parser.add_argument("--num-players", type=int, default=2,
                       help="Number of players")
    parser.add_argument("--dice-per-player", type=int, default=5,
                       help="Number of dice per player")
    parser.add_argument("--ones-wild", action="store_true",
                       help="Enable ones_wild rule")
    
    # Training options
    parser.add_argument("--win-rate-threshold", type=float, default=None,
                       help=f"Win rate threshold (default: {EXPERT_CONFIG['win_rate_threshold']:.1%})")
    parser.add_argument("--fresh-start", action="store_true",
                       help="Start training from scratch (ignore base PPO model)")
    
    # Classifier-specific options
    parser.add_argument("--classifier-samples", type=int, default=1000,
                       help="Number of game trajectories per opponent for classifier training")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device for neural network training")
    
    args = parser.parse_args()
    
    # Setup game configuration
    game_config = GameConfig(
        num_players=args.num_players,
        total_dice=args.dice_per_player,
        faces=(1, 2, 3, 4, 5, 6),
        ones_wild=args.ones_wild
    )
    
    # Get opponent classes
    opponent_classes = {}
    for agent_name, agent_cls in AGENT_MAP.items():
        if agent_name not in ["rl_ppo", "adaptive"]:  # Exclude RL agents
            try:
                _ = agent_cls()
                opponent_classes[agent_cls.__name__] = agent_cls
            except Exception:
                pass
    
    # === TRAIN ALL MODE ===
    if args.train_all:
        timesteps = args.timesteps or EXPERT_CONFIG["timesteps"]
        win_rate = args.win_rate_threshold or EXPERT_CONFIG["win_rate_threshold"]
        
        train_adaptive_system(
            game_config=game_config,
            opponent_classes=opponent_classes,
            base_model_path=None if args.fresh_start else PATH_CONFIG["generalist_model"],
            timesteps_per_expert=timesteps,
            classifier_samples=args.classifier_samples,
            win_rate_threshold=win_rate
        )
        return
    
    # === CLASSIFIER TRAINING MODE ===
    if args.train_classifier:
        train_neural_classifier(
            opponent_classes=opponent_classes,
            game_config=game_config,
            samples_per_opponent=args.classifier_samples,
            save_path=None,  # Uses PATH_CONFIG default
            device=args.device
        )
        return
    
    # === EXPERT EVALUATION MODE ===
    if args.evaluate_experts:
        evaluate_specialist_experts(
            opponent_classes=opponent_classes,
            game_config=game_config,
            num_games=100
        )
        return
    
    # === SINGLE EXPERT TRAINING MODE ===
    if args.opponent:
        opponent_cls = OPPONENTS[args.opponent]
        opponent_name = opponent_cls.__name__
        print(f"\nTraining single expert for adaptive system: {opponent_name}\n")
        
        base_model_path = PATH_CONFIG["generalist_model"]
        timesteps = args.timesteps or EXPERT_CONFIG["timesteps"]
        win_rate = args.win_rate_threshold or EXPERT_CONFIG["win_rate_threshold"]
        
        expert_path = train_specialist_expert(
            opponent_cls=opponent_cls,
            opponent_name=opponent_name,
            game_config=game_config,
            base_model_path=base_model_path if not args.fresh_start and os.path.exists(base_model_path + ".zip") else None,
            total_timesteps=timesteps,
            win_rate_threshold=win_rate
        )
        
        print(f"\n{'='*80}")
        print(f"Expert training complete: {expert_path}.zip")
        print(f"{'='*80}\n")
        return
    
    # No mode specified
    parser.error("Must specify --train-all, --train-classifier, --evaluate-experts, or --opponent")


if __name__ == "__main__":
    main()
