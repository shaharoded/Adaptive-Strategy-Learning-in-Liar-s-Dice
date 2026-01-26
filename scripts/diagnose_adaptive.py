"""
Unified Diagnostic Tool for Adaptive Agent

Tests adaptive agent performance against selected adversaries and compares
with direct expert performance. Shows opponent identification accuracy,
confidence, and win rates by player position.

Usage:
    python scripts/diagnose_adaptive.py                    # Test against all opponents
    python scripts/diagnose_adaptive.py GeneralistPPO      # Test against specific opponent
    python scripts/diagnose_adaptive.py --list             # List available opponents
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine
from liars_dice.agents.ppo_agent import PPOAgent
from liars_dice.agents.adaptive_agent import AdaptiveAgent
from liars_dice.agents.adaptive_agent_utils.action_tracker import ActionTrackerWrapper
from liars_dice.agents.adaptive_agent_utils.config import PATH_CONFIG
from liars_dice.agents import AGENT_MAP


def get_available_opponents():
    """Get list of available opponent types."""
    opponents = {}
    
    # Add all registered agents except PPO and adaptive
    for name, cls in AGENT_MAP.items():
        if name not in ["rl_ppo", "adaptive"]:
            try:
                _ = cls()
                opponents[cls.__name__] = cls
            except:
                pass
    
    # Add frozen generalist
    class GeneralistPPOAgent(PPOAgent):
        def __init__(self):
            super().__init__(model_path=PATH_CONFIG["generalist_model"], stochastic=False)
    
    opponents["GeneralistPPO"] = GeneralistPPOAgent
    
    return opponents


def play_match(agent1, agent2, config, agent1_idx=0):
    """
    Play a complete match with dice elimination.
    
    Returns:
        winner index (0 or 1)
    """
    engine = GameEngine(config)
    dice_counts = [config.total_dice, config.total_dice]
    agents = [agent1, agent2]
    
    while min(dice_counts) > 0:
        for i in range(2):
            engine.state.players[i].num_dice = dice_counts[i]
        engine.start_new_round()
        
        while not engine.is_terminal():
            current_player = engine.state.public.current_player
            view = engine.get_view(current_player)
            
            try:
                action = agents[current_player].choose_action(view)
                engine.apply_action(current_player, action)
            except Exception:
                break
        
        if engine.is_terminal():
            loser = engine.state.public.loser
            dice_counts[loser] -= 1
    
    return 0 if dice_counts[0] > 0 else 1


def test_direct_expert(opponent_name: str, opponent_cls, config: GameConfig, num_games: int = 2, verbose: bool = False):
    """Test specialist expert directly against opponent."""
    specialist_path = Path(PATH_CONFIG["adaptive_models_dir"]) / f"expert_{opponent_name}"
    
    try:
        # Use auto-sync mode (standalone) so expert can track opponent actions
        specialist = PPOAgent(model_path=str(specialist_path), stochastic=False, disable_auto_sync=False)
    except Exception as e:
        return None  # Expert doesn't exist
    
    results = {"p0_wins": 0, "p1_wins": 0, "games": num_games, "game_details": []}
    
    for game_num in range(num_games):
        specialist_is_p0 = (game_num % 2 == 0)
        opponent = opponent_cls()
        
        if specialist_is_p0:
            winner = play_match(specialist, opponent, config)
            won = (winner == 0)
            if won:
                results["p0_wins"] += 1
        else:
            winner = play_match(opponent, specialist, config)
            won = (winner == 1)
            if won:
                results["p1_wins"] += 1
        
        results["game_details"].append({
            "game_num": game_num + 1,
            "position": "P0" if specialist_is_p0 else "P1",
            "won": won
        })
        
        if verbose:
            pos = "P0" if specialist_is_p0 else "P1"
            result = "WIN" if won else "LOSS"
            print(f"  Game {game_num + 1}: {pos} - {result}")
        
        specialist.reset()
    
    results["total_wins"] = results["p0_wins"] + results["p1_wins"]
    results["win_rate"] = results["total_wins"] / num_games
    return results


def test_adaptive_agent(opponent_name: str, opponent_cls, config: GameConfig, num_games: int = 2, verbose: bool = False):
    """Test adaptive agent against opponent."""
    adaptive = AdaptiveAgent(device="cpu")
    
    results = {
        "p0_wins": 0,
        "p1_wins": 0,
        "games": num_games,
        "expert_switches": [],
        "identification_turns": [],
        "confidences": [],
        "correct_ids": 0,
        "game_details": []
    }
    
    for game_num in range(num_games):
        adaptive.reset()
        adaptive_is_p0 = (game_num % 2 == 0)
        
        opponent = opponent_cls()
        wrapped_opponent = ActionTrackerWrapper(opponent, adaptive_agent=adaptive)
        
        if adaptive_is_p0:
            agents = [adaptive, wrapped_opponent]
            adaptive_idx = 0
        else:
            agents = [wrapped_opponent, adaptive]
            adaptive_idx = 1
        
        # Track expert selection
        expert_used = None
        expert_turn = None
        final_confidence = 0.0
        turn_details = []
        
        engine = GameEngine(config)
        dice_counts = [config.total_dice, config.total_dice]
        turn_count = 0
        
        if verbose:
            pos = "P0" if adaptive_is_p0 else "P1"
            print(f"\n  Game {game_num + 1}: Adaptive as {pos}")
        
        while min(dice_counts) > 0:
            for i in range(2):
                engine.state.players[i].num_dice = dice_counts[i]
            engine.start_new_round()
            
            while not engine.is_terminal():
                current_player = engine.state.public.current_player
                view = engine.get_view(current_player)
                
                if current_player == adaptive_idx:
                    turn_count += 1
                    
                    # Get current beliefs and expert
                    beliefs = adaptive.belief_tracker.get_belief_distribution()
                    current_expert = adaptive.current_expert or "Generalist"
                    top_belief = max(beliefs.items(), key=lambda x: x[1]) if beliefs else (None, 0.0)
                    
                    # Check if expert switched
                    if expert_used is None and adaptive.current_expert is not None:
                        expert_used = adaptive.current_expert
                        expert_turn = turn_count
                        final_confidence = beliefs.get(expert_used, 0.0)
                        if verbose:
                            print(f"    Turn {turn_count}: Expert activated -> {expert_used} (conf: {final_confidence:.1%})")
                    
                    # Verbose: show top 3 predictions each turn
                    if verbose and turn_count <= 10:
                        if beliefs:
                            sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)[:3]
                            pred_str = ", ".join([f"{name}: {conf:.1%}" for name, conf in sorted_beliefs])
                            print(f"      Beliefs: {pred_str}")
                    
                    # Track turn details
                    turn_details.append({
                        "turn": turn_count,
                        "expert": current_expert,
                        "top_prediction": top_belief[0],
                        "confidence": top_belief[1]
                    })
                
                try:
                    action = agents[current_player].choose_action(view)
                    engine.apply_action(current_player, action)
                except Exception:
                    break
            
            if engine.is_terminal():
                loser = engine.state.public.loser
                dice_counts[loser] -= 1
        
        # Record results
        winner = 0 if dice_counts[0] > 0 else 1
        won = (winner == adaptive_idx)
        if won:
            if adaptive_is_p0:
                results["p0_wins"] += 1
            else:
                results["p1_wins"] += 1
        
        # Get final prediction
        beliefs = adaptive.belief_tracker.get_belief_distribution()
        predicted = max(beliefs.items(), key=lambda x: x[1])[0] if beliefs else None
        correct_id = (predicted == opponent_name)
        
        results["expert_switches"].append(expert_used or "Generalist")
        results["identification_turns"].append(expert_turn or turn_count)
        results["confidences"].append(beliefs.get(predicted, 0.0) if predicted else 0.0)
        if correct_id:
            results["correct_ids"] += 1
        
        # Store detailed game info
        results["game_details"].append({
            "game_num": game_num + 1,
            "position": "P0" if adaptive_is_p0 else "P1",
            "won": won,
            "expert_used": expert_used or "Generalist",
            "expert_turn": expert_turn or turn_count,
            "predicted": predicted,
            "correct_id": correct_id,
            "final_confidence": beliefs.get(predicted, 0.0) if predicted else 0.0,
            "turn_count": turn_count,
            "turn_details": turn_details
        })
        
        if verbose:
            result = "WIN" if won else "LOSS"
            expert_str = expert_used or "Generalist"
            id_str = f"ID: {predicted} ({'✓' if correct_id else '✗'})" if predicted else "ID: None"
            print(f"    Result: {result} | Expert: {expert_str} | {id_str}")
    
    results["total_wins"] = results["p0_wins"] + results["p1_wins"]
    results["win_rate"] = results["total_wins"] / num_games
    results["id_accuracy"] = results["correct_ids"] / num_games
    results["avg_confidence"] = sum(results["confidences"]) / len(results["confidences"])
    results["avg_id_turn"] = sum(results["identification_turns"]) / len(results["identification_turns"])
    
    return results


def print_report(opponent_name: str, adaptive_results: Dict, expert_results: Dict = None, verbose: bool = False):
    """Print formatted diagnostic report."""
    print("\n" + "="*80)
    print(f"DIAGNOSTIC REPORT: {opponent_name}")
    print("="*80)
    
    print("\n[ADAPTIVE AGENT]")
    print(f"  Win Rate:          {adaptive_results['win_rate']:.1%} ({adaptive_results['total_wins']}/{adaptive_results['games']})")
    print(f"  P0 Win Rate:       {adaptive_results['p0_wins']/(adaptive_results['games']//2):.1%}")
    print(f"  P1 Win Rate:       {adaptive_results['p1_wins']/(adaptive_results['games']//2):.1%}")
    print(f"  ID Accuracy:       {adaptive_results['id_accuracy']:.1%}")
    print(f"  Avg Confidence:    {adaptive_results['avg_confidence']:.1%}")
    print(f"  Avg ID Turn:       {adaptive_results['avg_id_turn']:.1f}")
    
    if verbose:
        print("\n  Game-by-Game Results:")
        for game in adaptive_results['game_details']:
            result = "WIN " if game['won'] else "LOSS"
            id_mark = "✓" if game['correct_id'] else "✗"
            print(f"    Game {game['game_num']}: {game['position']} {result} | Expert: {game['expert_used']:20s} | ID: {game['predicted'] or 'None':20s} [{id_mark}] | Conf: {game['final_confidence']:5.1%} | Turn {game['expert_turn']}/{game['turn_count']}")
    
    if expert_results:
        print("\n[DIRECT EXPERT]")
        print(f"  Win Rate:          {expert_results['win_rate']:.1%} ({expert_results['total_wins']}/{expert_results['games']})")
        print(f"  P0 Win Rate:       {expert_results['p0_wins']/(expert_results['games']//2):.1%}")
        print(f"  P1 Win Rate:       {expert_results['p1_wins']/(expert_results['games']//2):.1%}")
        
        if verbose:
            print("\n  Game-by-Game Results:")
            for game in expert_results['game_details']:
                result = "WIN " if game['won'] else "LOSS"
                print(f"    Game {game['game_num']}: {game['position']} {result}")
        
        # Comparison
        delta = adaptive_results['win_rate'] - expert_results['win_rate']
        print("\n[COMPARISON]")
        print(f"  Delta:             {delta:+.1%}")
        
        if abs(delta) < 0.1:
            status = "✓ Performance comparable"
        elif delta > 0:
            status = "✓ Adaptive outperforms"
        else:
            status = "⚠ Performance gap detected"
        print(f"  Status:            {status}")
    else:
        print("\n[DIRECT EXPERT]")
        print(f"  Status:            Expert not trained for this opponent")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Adaptive Agent Performance")
    parser.add_argument("opponent", nargs="?", help="Opponent name to test against")
    parser.add_argument("--list", action="store_true", help="List available opponents")
    parser.add_argument("--games", type=int, default=10, help="Number of games per test")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed turn-by-turn information")
    args = parser.parse_args()
    
    opponents = get_available_opponents()
    
    if args.list:
        print("\nAvailable Opponents:")
        for name in sorted(opponents.keys()):
            print(f"  - {name}")
        return
    
    config = GameConfig(num_players=2, total_dice=5, faces=(1,2,3,4,5,6), ones_wild=False)
    
    if args.opponent:
        # Test single opponent
        if args.opponent not in opponents:
            print(f"Error: Opponent '{args.opponent}' not found. Use --list to see available opponents.")
            return
        
        print(f"\nTesting against {args.opponent}...")
        opponent_cls = opponents[args.opponent]
        
        print("\n[Running adaptive agent tests...]")
        adaptive_results = test_adaptive_agent(args.opponent, opponent_cls, config, args.games, args.verbose)
        
        print("\n[Running direct expert tests...]")
        expert_results = test_direct_expert(args.opponent, opponent_cls, config, args.games, args.verbose)
        
        print_report(args.opponent, adaptive_results, expert_results, args.verbose)
    
    else:
        # Test all opponents
        print(f"\nTesting against {len(opponents)} opponents ({args.games} games each)...\n")
        
        summary = []
        for opp_name in sorted(opponents.keys()):
            print(f"Testing {opp_name}...", end=" ", flush=True)
            opp_cls = opponents[opp_name]
            
            adaptive_results = test_adaptive_agent(opp_name, opp_cls, config, args.games, verbose=False)
            expert_results = test_direct_expert(opp_name, opp_cls, config, args.games, verbose=False)
            
            summary.append({
                "name": opp_name,
                "adaptive_wr": adaptive_results["win_rate"],
                "expert_wr": expert_results["win_rate"] if expert_results else None,
                "id_accuracy": adaptive_results["id_accuracy"]
            })
            print("Done")
        
        # Print summary table
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        print(f"{'Opponent':<30} {'Adaptive WR':<15} {'Expert WR':<15} {'Delta':<15} {'ID Acc':<10}")
        print("-"*100)
        
        for row in summary:
            adaptive_str = f"{row['adaptive_wr']:.1%}"
            expert_str = f"{row['expert_wr']:.1%}" if row['expert_wr'] is not None else "N/A"
            delta_str = f"{row['adaptive_wr'] - row['expert_wr']:+.1%}" if row['expert_wr'] is not None else "N/A"
            id_str = f"{row['id_accuracy']:.1%}"
            print(f"{row['name']:<30} {adaptive_str:<15} {expert_str:<15} {delta_str:<15} {id_str:<10}")
        
        print("="*100)


if __name__ == "__main__":
    main()
