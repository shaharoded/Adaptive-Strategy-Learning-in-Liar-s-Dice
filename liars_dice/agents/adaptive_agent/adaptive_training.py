"""
adaptive_training.py

Training utilities for the adaptive agent system.
Handles training of specialist PPO experts and the neural opponent classifier.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine
from liars_dice.agents.ppo_agent import train_ppo_agent
from liars_dice.agents.adapter_agent.config import CLASSIFIER_CONFIG, EXPERT_CONFIG, PATH_CONFIG
from liars_dice.agents.reinforcement_agent.config import TRAINING_CONFIG
from liars_dice.agents.adapter_agent.neural_belief_tracker import NeuralBeliefTracker
from liars_dice.agents.adapter_agent.trajectory_encoder import TrajectoryEncoder


def train_specialist_expert(
    opponent_cls,
    opponent_name: str,
    game_config: GameConfig,
    base_model_path: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    win_rate_threshold: Optional[float] = None
) -> str:
    """
    Train a specialist PPO expert against a specific opponent.
    
    Args:
        opponent_cls: The opponent class to train against
        opponent_name: Name for saving the expert model
        game_config: Game configuration
        base_model_path: Optional path to base PPO model for warm start
        total_timesteps: Training timesteps (default: from EXPERT_CONFIG)
        win_rate_threshold: Target win rate (default: from EXPERT_CONFIG)
        
    Returns:
        Path to the saved expert model
    """
    # Use config defaults if not specified
    if total_timesteps is None:
        total_timesteps = EXPERT_CONFIG["timesteps"]
    if win_rate_threshold is None:
        win_rate_threshold = EXPERT_CONFIG["win_rate_threshold"]
    
    save_dir = Path(PATH_CONFIG["adaptive_models_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize opponent name for filename
    safe_name = opponent_name.replace(" ", "_").replace("/", "_")
    expert_name = f"expert_{safe_name}"
    
    print(f"\n{'='*80}")
    print(f"Training Specialist Expert: {opponent_name}")
    print(f"Target: {win_rate_threshold:.1%} win rate")
    print(f"{'='*80}\n")
    
    # Temporarily override the training config to use adaptive_agent log directory
    original_log_dir = TRAINING_CONFIG["log_dir"]
    TRAINING_CONFIG["log_dir"] = EXPERT_CONFIG["log_dir"]
    
    try:
        # Train using the existing PPO training infrastructure
        saved_path = train_ppo_agent(
            opponent_cls=opponent_cls,
            game_config=game_config,
            load_path=base_model_path,  # Warm start from base model if provided
            save_name=f"adaptive_models/{expert_name}",
            total_timesteps=total_timesteps,
            enable_early_stopping=True,
            win_rate_threshold=win_rate_threshold
        )
    finally:
        # Restore original log directory
        TRAINING_CONFIG["log_dir"] = original_log_dir
    
    print(f"\n✓ Expert trained successfully: {saved_path}.zip")
    return saved_path


class OpponentTrajectoryDataset(Dataset):
    """PyTorch Dataset for opponent trajectories."""
    
    def __init__(self, trajectories: List[Tuple[List, str]], opponent_types: List[str], max_sequence_length: int = 50):
        """
        Args:
            trajectories: List of (trajectory, opponent_name) tuples
            opponent_types: List of all opponent type names
            max_sequence_length: Maximum trajectory length for encoding
        """
        self.trajectories = trajectories
        self.opponent_types = opponent_types
        self.max_sequence_length = max_sequence_length
        self.opponent_to_idx = {name: idx for idx, name in enumerate(opponent_types)}
        self.encoder = TrajectoryEncoder(max_sequence_length=max_sequence_length)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory, opponent_name = self.trajectories[idx]
        
        # Encode trajectory
        features, mask = self.encoder.encode_trajectory(trajectory)
        
        # Get label
        label = self.opponent_to_idx[opponent_name]
        
        return {
            "features": torch.FloatTensor(features),
            "mask": torch.FloatTensor(mask),
            "label": torch.LongTensor([label])[0]
        }


def collect_opponent_trajectories(
    opponent_cls,
    opponent_name: str,
    game_config: GameConfig,
    num_games: int = 1000
) -> List[Tuple[List[Tuple], str]]:
    """
    Collect full game trajectories from an opponent for neural network training.
    
    Args:
        opponent_cls: The opponent class
        opponent_name: Name of the opponent type
        game_config: Game configuration
        num_games: Number of games to collect
        
    Returns:
        List of (trajectory, opponent_name) tuples where trajectory is a list of
        (action, player_id, game_state, revealed_dice) tuples
    """
    print(f"Collecting trajectories from {opponent_name}... ({num_games} games)", flush=True)
    
    from liars_dice.agents.random_agent import RandomAgent
    player = RandomAgent()
    
    game_trajectories = []
    
    for game_idx in range(num_games):
        opponent = opponent_cls()
        
        # Alternate starting player
        if game_idx % 2 == 0:
            agents = [player, opponent]
            opponent_player_id = 1
        else:
            agents = [opponent, player]
            opponent_player_id = 0
        
        # Track full game trajectory
        game_trajectory = []
        
        # Play match with dice elimination
        engine = GameEngine(game_config)
        dice_counts = [game_config.total_dice, game_config.total_dice]
        
        while min(dice_counts) > 0:
            # Update dice counts
            for i in range(2):
                engine.state.players[i].num_dice = dice_counts[i]
            engine.start_new_round()
            
            # Track actions in this round
            round_actions = []
            
            # Play round
            while not engine.is_terminal():
                current_player = engine.state.public.current_player
                view = engine.get_view(current_player)
                action = agents[current_player].choose_action(view)
                
                # Extract game state
                public = view["public"]
                game_state = {
                    "last_bid": public.last_bid,
                    "total_dice": sum(public.dice_counts),
                    "my_dice_count": len(view.get("my_dice", [])),
                    "opp_dice_count": sum(c for i, c in enumerate(public.dice_counts) if i != current_player),
                    "round_index": public.round_index
                }
                
                # Record action (both players, but we'll filter for opponent later)
                round_actions.append((action, current_player, game_state, None))
                
                try:
                    engine.apply_action(current_player, action)
                except Exception:
                    break
            
            # Round ended - add revealed dice to last action
            if engine.is_terminal() and round_actions:
                # Get revealed dice from both players
                revealed_dice = []
                for player_idx in range(2):
                    revealed_dice.extend(engine.state.players[player_idx].private_dice)
                
                # Update last action with revealed dice
                last_action = round_actions[-1]
                round_actions[-1] = (last_action[0], last_action[1], last_action[2], revealed_dice)
            
            # Add round actions to game trajectory
            game_trajectory.extend(round_actions)
            
            # Update dice counts
            if engine.is_terminal():
                round_loser = engine.state.public.loser
                dice_counts[round_loser] -= 1
        
        # Filter for opponent actions only
        opponent_trajectory = [
            (action, player_id, game_state, revealed_dice)
            for action, player_id, game_state, revealed_dice in game_trajectory
            if player_id == opponent_player_id
        ]
        
        if opponent_trajectory:  # Only add if opponent made at least one action
            game_trajectories.append((opponent_trajectory, opponent_name))
        
        # Progress
        if (game_idx + 1) % 100 == 0:
            print(f"  Collected {game_idx + 1}/{num_games} games...", flush=True)
    
    print(f"✓ Collected {len(game_trajectories)} game trajectories from {opponent_name}\n", flush=True)
    return game_trajectories


def train_neural_classifier(
    opponent_classes: Dict[str, type],
    game_config: GameConfig,
    samples_per_opponent: int = 1000,
    save_path: Optional[str] = None,
    device: str = "cpu"
) -> NeuralBeliefTracker:
    """
    Train the neural LSTM-based opponent classifier.
    
    Args:
        opponent_classes: Dict mapping opponent names to their classes
        game_config: Game configuration
        samples_per_opponent: Number of game trajectories to collect per opponent
        save_path: Path to save trained model (default: from PATH_CONFIG)
        device: torch device ("cpu" or "cuda")
        
    Returns:
        Trained NeuralBeliefTracker
    """
    # Use config default if not specified
    if save_path is None:
        save_path = PATH_CONFIG["neural_classifier"]
    
    # Resolve path relative to this module's directory (liars_dice/agents/adapter_agent/)
    save_path = Path(save_path)
    if not save_path.is_absolute():
        save_path = (Path(__file__).parent.parent / save_path).resolve()
    save_path = str(save_path)
    
    print("\n" + "="*80)
    print("TRAINING NEURAL OPPONENT CLASSIFIER")
    print("="*80 + "\n")
    
    opponent_names = list(opponent_classes.keys())
    
    # Phase 1: Collect trajectories
    print(f"Phase 1: Collecting Trajectories from {len(opponent_names)} Opponents")
    print(f"  Games per opponent: {samples_per_opponent}")
    print("="*80 + "\n")
    
    all_trajectories = []
    for opp_name, opp_cls in opponent_classes.items():
        try:
            trajectories = collect_opponent_trajectories(
                opp_cls,
                opp_name,
                game_config,
                num_games=samples_per_opponent
            )
            all_trajectories.extend(trajectories)
        except Exception as e:
            print(f"⚠️  Warning: Failed to collect from {opp_name}: {e}")
            continue
    
    if not all_trajectories:
        raise ValueError("No trajectories collected. Cannot train classifier.")
    
    print(f"\n✓ Total trajectories collected: {len(all_trajectories)}")
    print(f"  Opponent types: {opponent_names}\n")
    
    # Phase 2: Train neural network
    print("="*80)
    print(f"Phase 2: Training LSTM Classifier")
    print(f"  Architecture: {CLASSIFIER_CONFIG['num_lstm_layers']}-layer LSTM")
    print(f"  Hidden dim: {CLASSIFIER_CONFIG['hidden_dim']}")
    print(f"  Epochs: {CLASSIFIER_CONFIG['num_epochs']}")
    print("="*80 + "\n")
    
    # Create dataset and dataloaders
    dataset = OpponentTrajectoryDataset(
        all_trajectories,
        opponent_names,
        max_sequence_length=CLASSIFIER_CONFIG["max_sequence_length"]
    )
    
    # Split train/val
    train_size = int(CLASSIFIER_CONFIG["train_val_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    tracker = NeuralBeliefTracker(opponent_names, device=device)
    tracker.train_mode()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        tracker.model.parameters(),
        lr=CLASSIFIER_CONFIG["learning_rate"],
        weight_decay=CLASSIFIER_CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # TensorBoard writer for monitoring
    tensorboard_dir = Path(PATH_CONFIG["tensorboard_log_dir"])
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    # Training loop with convergence monitoring
    best_val_loss = float('inf')
    patience_counter = 0
    min_improvement = CLASSIFIER_CONFIG["min_loss_improvement"]
    
    print(f"TensorBoard: tensorboard --logdir={PATH_CONFIG['tensorboard_log_dir']}\n")
    
    for epoch in range(CLASSIFIER_CONFIG["num_epochs"]):
        # Train
        tracker.train_mode()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CLASSIFIER_CONFIG['num_epochs']}"):
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits, _ = tracker.model(features, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validate
        tracker.eval_mode()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)
                
                logits, _ = tracker.model(features, mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log loss improvement for convergence monitoring
        loss_improvement = best_val_loss - val_loss
        writer.add_scalar('Convergence/loss_improvement', loss_improvement, epoch)
        writer.add_scalar('Convergence/patience_counter', patience_counter, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping with convergence monitoring
        if val_loss < best_val_loss - min_improvement:
            # Significant improvement
            improvement_pct = 100 * loss_improvement / best_val_loss if best_val_loss > 0 else 0
            print(f"  → Improvement: {loss_improvement:.4f} ({improvement_pct:.2f}%)")
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tracker.save_model(save_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            # No significant improvement
            patience_counter += 1
            print(f"  → No significant improvement (patience: {patience_counter}/{CLASSIFIER_CONFIG['early_stopping_patience']})")
            if patience_counter >= CLASSIFIER_CONFIG["early_stopping_patience"]:
                print(f"\nEarly stopping: Loss converged (no improvement > {min_improvement:.4f} for {patience_counter} epochs)")
                writer.add_text('Training/early_stop_reason', f'Loss converged at epoch {epoch+1}', epoch)
                break
    
    # Close TensorBoard writer
    writer.close()
    
    # Load best model
    tracker.load_model(save_path)
    tracker.eval_mode()
    
    print(f"\n✓ Classifier training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {save_path}")
    print(f"  TensorBoard logs: {PATH_CONFIG['tensorboard_log_dir']}\n")
    
    return tracker


def evaluate_specialist_experts(
    opponent_classes: Dict[str, type],
    game_config: GameConfig,
    num_games: int = 100
) -> Dict[str, Tuple[int, int, float]]:
    """
    Evaluate all specialist experts against their target opponents.
    
    Args:
        opponent_classes: Dict mapping opponent names to their classes
        game_config: Game configuration
        num_games: Number of evaluation games per expert
        
    Returns:
        Dict mapping opponent names to (wins, total, win_rate) tuples
    """
    from liars_dice.agents.ppo_agent import PPOAgent
    
    print("\n" + "="*80)
    print("EVALUATING SPECIALIST EXPERTS")
    print("="*80 + "\n")
    
    results = {}
    
    for opp_name, opp_cls in opponent_classes.items():
        # Load specialist expert
        safe_name = opp_name.replace(" ", "_").replace("/", "_")
        expert_path = f"{PATH_CONFIG['expert_prefix']}_{safe_name}"
        
        try:
            expert = PPOAgent(model_path=expert_path)
        except Exception as e:
            print(f"⚠️  Could not load expert for {opp_name}: {e}")
            continue
        
        # Evaluate
        wins = 0
        for game_idx in range(num_games):
            opponent = opp_cls()
            
            # Alternate who goes first
            if game_idx % 2 == 0:
                agents = [expert, opponent]
                expert_player = 0
            else:
                agents = [opponent, expert]
                expert_player = 1
            
            # Play match with dice elimination
            engine = GameEngine(game_config)
            dice_counts = [game_config.total_dice, game_config.total_dice]
            
            while min(dice_counts) > 0:
                for i in range(2):
                    engine.state.players[i].num_dice = dice_counts[i]
                engine.start_new_round()
                
                while not engine.is_terminal():
                    current_player = engine.state.public.current_player
                    view = engine.get_view(current_player)
                    action = agents[current_player].choose_action(view)
                    
                    try:
                        engine.apply_action(current_player, action)
                    except Exception:
                        winner = 1 - current_player
                        break
                
                if engine.is_terminal():
                    round_loser = engine.state.public.loser
                    dice_counts[round_loser] -= 1
            
            # Check winner
            winner = 0 if dice_counts[0] > 0 else 1
            if winner == expert_player:
                wins += 1
        
        win_rate = wins / num_games
        results[opp_name] = (wins, num_games, win_rate)
        
        # Display result
        bar_length = int(win_rate * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        marker = "✓" if win_rate >= 0.99 else "⚠️"
        
        print(f"{marker} {opp_name:<25} {win_rate:>6.1%} {bar}  {wins}/{num_games}")
    
    print("\n" + "="*80 + "\n")
    return results


def load_neural_classifier(
    classifier_path: Optional[str] = None,
    device: str = "cpu"
) -> NeuralBeliefTracker:
    """
    Load a trained neural classifier from disk.
    
    Args:
        classifier_path: Path to the saved neural classifier model (default: from PATH_CONFIG)
        device: torch device
        
    Returns:
        NeuralBeliefTracker with loaded model
    """
    # Use config default if not specified
    if classifier_path is None:
        classifier_path = PATH_CONFIG["neural_classifier"]
    
    # Resolve path relative to agents directory
    classifier_path = Path(classifier_path)
    if not classifier_path.is_absolute():
        classifier_path = (Path(__file__).parent.parent / classifier_path).resolve()
    classifier_path = str(classifier_path)
    
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Neural classifier not found at {classifier_path}")
    
    # Load checkpoint to get opponent types
    checkpoint = torch.load(classifier_path, map_location=torch.device(device))
    opponent_types = checkpoint["opponent_types"]
    
    # Create and load tracker
    tracker = NeuralBeliefTracker(opponent_types, model_path=classifier_path, device=device)
    tracker.eval_mode()
    
    return tracker

