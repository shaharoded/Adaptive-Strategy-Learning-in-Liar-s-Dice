"""
neural_belief_tracker.py

LSTM-based neural network for opponent identification.
Learns to predict opponent type from action trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

from liars_dice.agents.adaptive_agent_utils.config import CLASSIFIER_CONFIG
from liars_dice.agents.adaptive_agent_utils.trajectory_encoder import TrajectoryEncoder


class OpponentClassifierLSTM(nn.Module):
    """
    LSTM-based classifier for opponent identification.
    
    Architecture:
    - Input: Trajectory features [batch, seq_len, feature_dim]
    - LSTM layers: Process sequence
    - Attention: Weight important timesteps
    - Output: Probability distribution over opponent types
    """
    
    def __init__(self, num_opponent_types: int, config: Dict = None):
        super().__init__()
        
        self.config = config or CLASSIFIER_CONFIG
        self.num_opponent_types = num_opponent_types
        
        # Dimensions
        self.feature_dim = 12  # From trajectory encoder
        self.hidden_dim = self.config["hidden_dim"]
        self.num_layers = self.config["num_lstm_layers"]
        
        # Input projection (optional)
        self.input_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.config["dropout"] if self.num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Attention mechanism (learn which actions are most informative)
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(self.hidden_dim // 2, num_opponent_types)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: [batch, seq_len, feature_dim] action features
            mask: [batch, seq_len] binary mask (1 = valid, 0 = padding)
            
        Returns:
            logits: [batch, num_opponent_types] unnormalized scores
            attention_weights: [batch, seq_len] attention weights
        """
        batch_size, seq_len, _ = features.shape
        
        # Project input features
        x = self.input_projection(features)  # [batch, seq_len, hidden_dim]
        x = self.layer_norm(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_dim]
        
        # Compute attention weights
        attention_scores = self.attention(lstm_out).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask to attention scores if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]
        
        # Weighted sum of LSTM outputs
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        context = (lstm_out * attention_weights_expanded).sum(dim=1)  # [batch, hidden_dim]
        
        # Output prediction
        logits = self.output_layers(context)  # [batch, num_opponent_types]
        
        return logits, attention_weights
    
    def predict_proba(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict probability distribution over opponent types.
        
        Returns:
            probs: [batch, num_opponent_types] probabilities
        """
        with torch.no_grad():
            logits, _ = self.forward(features, mask)
            probs = F.softmax(logits, dim=-1)
        return probs


class NeuralBeliefTracker:
    """
    Neural network-based belief tracker for opponent identification.
    Maintains and updates beliefs using an LSTM classifier.
    """
    
    def __init__(
        self,
        opponent_types: List[str],
        model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            opponent_types: List of opponent type names
            model_path: Path to trained model (if available)
            device: torch device ("cpu" or "cuda")
        """
        self.opponent_types = opponent_types
        self.num_types = len(opponent_types)
        self.device = torch.device(device)
        
        # Initialize encoder
        self.encoder = TrajectoryEncoder()
        
        # Initialize model
        self.model = OpponentClassifierLSTM(
            num_opponent_types=self.num_types,
            config=CLASSIFIER_CONFIG
        ).to(self.device)
        
        # Load trained model if available
        if model_path:
            self.load_model(model_path)
        
        # Current trajectory buffer
        self.trajectory = []
        
        # Current belief distribution (updated incrementally)
        self.beliefs = np.ones(self.num_types) / self.num_types
        self.prediction_history = []
    
    def reset(self):
        """Reset trajectory and beliefs for new game."""
        self.trajectory = []
        self.beliefs = np.ones(self.num_types) / self.num_types
        self.prediction_history = []
    
    def update_belief(
        self,
        action,
        player_id: int,
        game_state: Dict,
        revealed_dice: Optional[List[int]] = None
    ):
        """
        Update beliefs based on observed action.
        
        Args:
            action: Opponent's action (BidAction or CallLiarAction)
            player_id: 0 = me, 1 = opponent
            game_state: Game state context
            revealed_dice: Revealed dice if round ended
        """
        # Add to trajectory
        self.trajectory.append((action, player_id, game_state, revealed_dice))
        
        # Encode trajectory
        features, mask = self.encoder.encode_trajectory(self.trajectory)
        
        # Convert to torch tensors
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)  # [1, seq_len, feat_dim]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)  # [1, seq_len]
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(features_tensor, mask_tensor)
            probs_np = probs.cpu().numpy()[0]  # [num_types]
        
        # Update beliefs (exponential moving average for stability)
        alpha = 0.3  # Smoothing factor (lower = more stable)
        self.beliefs = alpha * probs_np + (1 - alpha) * self.beliefs
        
        # Renormalize
        self.beliefs = self.beliefs / np.sum(self.beliefs)
        
        # Store prediction
        self.prediction_history.append(probs_np.copy())
    
    def get_best_opponent(self, confidence_threshold: float = 0.7) -> Optional[str]:
        """
        Get most likely opponent if confidence exceeds threshold.
        
        Args:
            confidence_threshold: Minimum probability to commit
            
        Returns:
            Opponent type name if confident, else None
        """
        max_idx = np.argmax(self.beliefs)
        max_prob = self.beliefs[max_idx]
        
        if max_prob >= confidence_threshold:
            return self.opponent_types[max_idx]
        return None
    
    def get_belief_distribution(self) -> Dict[str, float]:
        """Get current belief distribution."""
        return {opp_type: float(prob) 
                for opp_type, prob in zip(self.opponent_types, self.beliefs)}
    
    def get_entropy(self) -> float:
        """Compute Shannon entropy of belief distribution."""
        beliefs_safe = np.clip(self.beliefs, 1e-10, 1.0)
        entropy = -np.sum(self.beliefs * np.log(beliefs_safe))
        return float(entropy)
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """
        Get attention weights from last prediction (which actions were important).
        
        Returns:
            Attention weights [seq_len] or None if no predictions yet
        """
        if not self.trajectory:
            return None
        
        # Encode current trajectory
        features, mask = self.encoder.encode_trajectory(self.trajectory)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model.forward(features_tensor, mask_tensor)
            weights_np = attention_weights.cpu().numpy()[0]
        
        return weights_np
    
    def load_model(self, model_path: str):
        """Load trained model from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    def save_model(self, model_path: str):
        """Save model to disk."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "opponent_types": self.opponent_types,
            "config": CLASSIFIER_CONFIG
        }, model_path)
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
