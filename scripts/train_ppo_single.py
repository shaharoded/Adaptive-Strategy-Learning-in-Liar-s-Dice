"""
train_ppo_single.py

Train PPO agent against a single opponent type.
Useful for focused training or resuming training against specific opponents.

Usage:
    python scripts/train_ppo_single.py --opponent random --timesteps 100000
    python scripts/train_ppo_single.py --opponent bayesian --resume --timesteps 50000
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from liars_dice.core.config import GameConfig
from liars_dice.agents.ppo_agent import train_ppo_agent
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


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent against a single opponent")
    parser.add_argument("--opponent", type=str, required=True, choices=OPPONENTS.keys(),
                       help="Opponent agent type")
    parser.add_argument("--timesteps", type=int, default=100_000,
                       help="Number of training timesteps")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from existing checkpoint")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Custom model name (default: ppo_model_<opponent>)")
    parser.add_argument("--num-players", type=int, default=2,
                       help="Number of players")
    parser.add_argument("--dice-per-player", type=int, default=5,
                       help="Number of dice per player")
    parser.add_argument("--ones-wild", action="store_true",
                       help="Enable ones_wild rule")
    
    args = parser.parse_args()
    
    # Get opponent class
    opponent_cls = OPPONENTS[args.opponent]
    
    # Setup model name
    model_name = args.model_name or f"ppo_model_{args.opponent}"
    
    # Setup game configuration
    game_config = GameConfig(
        num_players=args.num_players,
        dice_per_player=args.dice_per_player,
        faces=(1, 2, 3, 4, 5, 6),
        ones_wild=args.ones_wild
    )
    
    # Determine load path for resuming
    load_path = None
    if args.resume:
        base_path = f"./liars_dice/agents/weights/{model_name}"
        if os.path.exists(base_path + ".zip"):
            load_path = base_path
            print(f"Resuming training from: {load_path}.zip")
        else:
            print(f"Warning: Resume requested but model not found at {base_path}.zip")
            print("Starting fresh training...")
    
    print("\n" + "="*80)
    print("PPO TRAINING - SINGLE OPPONENT")
    print("="*80 + "\n")
    print(f"Configuration:")
    print(f"  Opponent: {args.opponent} ({opponent_cls.__name__})")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Game: {args.num_players} players, {args.dice_per_player} dice each")
    print(f"  Ones Wild: {args.ones_wild}")
    print(f"  Model Name: {model_name}")
    print(f"  Resume: {args.resume}")
    print()
    
    # Train
    saved_path = train_ppo_agent(
        opponent_cls=opponent_cls,
        game_config=game_config,
        load_path=load_path,
        save_name=model_name,
        total_timesteps=args.timesteps,
        log_interval=10
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80 + "\n")
    print(f"Model saved to: {saved_path}.zip")
    print("\nTo visualize training progress:")
    print("  tensorboard --logdir=./runs/ppo_training/")
    print("\nTo resume training:")
    print(f"  python scripts/train_ppo_single.py --opponent {args.opponent} --resume --timesteps <more_steps>")
    print()


if __name__ == "__main__":
    main()
