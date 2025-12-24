"""
Script to train and save NashCFRAgent CFR policies for all dice count combinations.
Saves to liars_dice/agents/weights/nash_cfr_policy.pkl by default.
"""
import os
import argparse
from liars_dice.agents.nash_agent import NashCFRAgent

def main():
    parser = argparse.ArgumentParser(description="Train NashCFRAgent CFR policies.")
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--max_dice', type=int, default=5, help='Max dice per player')
    parser.add_argument('--faces', type=int, nargs='+', default=[1,2,3,4,5,6], help='Dice faces')
    parser.add_argument('--iterations', type=int, default=10000, help='CFR iterations per policy')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file to resume/save')
    parser.add_argument('--tensorboard', type=str, default=None, help='TensorBoard log directory')
    args = parser.parse_args()

    print(f"Training NashCFRAgent CFR policies for {args.num_players} players, dice counts 1-{args.max_dice}, faces={args.faces}, iterations={args.iterations}")
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'liars_dice', 'agents', 'weights')
    weights_path = os.path.join(weights_dir, 'nash_cfr_policy.pkl')
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint_path = args.checkpoint if args.checkpoint else weights_path
    policies = NashCFRAgent.train_multi_policy(
        num_players=args.num_players,
        max_dice=args.max_dice,
        faces=tuple(args.faces),
        iterations=args.iterations,
        verbose=True,
        checkpoint_path=checkpoint_path,
        tensorboard_logdir=args.tensorboard,
    )
    NashCFRAgent.save_policy_dict(policies, weights_path)
    print(f"Policies saved to {weights_path}")

if __name__ == "__main__":
    main()
