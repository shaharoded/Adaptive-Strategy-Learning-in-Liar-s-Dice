"""
trajectory_encoder.py

Encoder for game trajectories that structures actions, revealed dice,
and game state into token sequences for the LSTM-based opponent classifier.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid
from liars_dice.agents.adapter_agent.config import ENCODER_CONFIG


class TrajectoryEncoder:
    """
    Encodes game trajectories into token sequences for neural network input.
    
    Each timestep includes:
    - Action type (bid/call_liar)
    - Bid details (quantity, face)
    - Player ID (me vs opponent)
    - Relative bid increase
    - Estimated probability
    - Revealed dice (if available)
    - Special tokens (start_round, end_round)
    """
    
    def __init__(self, max_sequence_length: int = 50):
        self.config = ENCODER_CONFIG
        self.max_sequence_length = max_sequence_length
        
        # Token vocabulary
        self.special_tokens = self.config["special_tokens"]
        self.vocab_size = self._compute_vocab_size()
        
        # Feature dimensions
        self.feature_dim = 12  # Number of features per timestep
        
    def _compute_vocab_size(self) -> int:
        """
        Compute vocabulary size for token embeddings.
        
        Tokens include:
        - Special tokens (PAD, START_ROUND, END_ROUND, CALL_LIAR)
        - Bid tokens: quantity (1-30) × face (1-6) = 180
        - Player tokens: 2 (me, opponent)
        Total: ~200 tokens
        """
        num_special = len(self.special_tokens)
        num_bid_tokens = self.config["max_quantity"] * self.config["num_faces"]
        num_player_tokens = 2
        return num_special + num_bid_tokens + num_player_tokens + 10  # +10 buffer
    
    def encode_action(
        self,
        action,
        player_id: int,  # 0 = me, 1 = opponent
        game_state: Dict,
        revealed_dice: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Encode a single action into a feature vector.
        
        Args:
            action: BidAction or CallLiarAction
            player_id: 0 for me, 1 for opponent
            game_state: Dict with game context (last_bid, total_dice, etc.)
            revealed_dice: List of revealed dice (if round ended)
            
        Returns:
            Feature vector [12 dimensions]:
            [action_type, bid_quantity_norm, bid_face_norm, player_id,
             quantity_increase, face_increase, is_aggressive, is_conservative,
             bid_probability, revealed_dice_match, revealed_dice_count, round_ended]
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        last_bid = game_state.get("last_bid")
        total_dice = game_state.get("total_dice", 10)
        
        # Player ID (0 or 1)
        features[3] = float(player_id)
        
        # Round ended flag
        features[11] = 1.0 if revealed_dice is not None else 0.0
        
        if isinstance(action, CallLiarAction):
            # Call liar action
            features[0] = 1.0  # action_type = call_liar
            
            # If we have revealed dice, compute match count
            if revealed_dice is not None and last_bid is not None:
                dice_counter = Counter(revealed_dice)
                match_count = dice_counter.get(last_bid.face, 0)
                features[9] = match_count / len(revealed_dice) if revealed_dice else 0.0
                features[10] = float(match_count)
            
        elif isinstance(action, BidAction):
            bid = action.bid
            
            # Action type (0 for bid)
            features[0] = 0.0
            
            # Normalized bid values
            features[1] = bid.quantity / self.config["max_quantity"]
            features[2] = bid.face / self.config["num_faces"]
            
            # Compute relative increase
            if last_bid is not None:
                quantity_increase = (bid.quantity - last_bid.quantity) / total_dice
                face_increase = (bid.face - last_bid.face) / self.config["num_faces"]
                features[4] = max(0.0, quantity_increase)
                features[5] = face_increase
                
                # Aggressiveness classification
                if bid.quantity > last_bid.quantity + 1:
                    features[6] = 1.0  # is_aggressive
                elif bid.quantity == last_bid.quantity and bid.face == last_bid.face + 1:
                    features[7] = 1.0  # is_conservative
            else:
                # Opening bid
                expected_dice = total_dice / self.config["num_faces"]
                if bid.quantity > expected_dice * 1.2:
                    features[6] = 1.0  # aggressive opening
                else:
                    features[7] = 1.0  # conservative opening
            
            # Estimate bid probability
            features[8] = self._estimate_bid_probability(bid, total_dice)
            
            # If revealed dice available, check match
            if revealed_dice is not None:
                dice_counter = Counter(revealed_dice)
                match_count = dice_counter.get(bid.face, 0)
                features[9] = match_count / len(revealed_dice) if revealed_dice else 0.0
                features[10] = float(match_count)
        
        return features
    
    def _estimate_bid_probability(self, bid: Bid, total_dice: int) -> float:
        """
        Estimate probability that a bid is true using binomial approximation.
        """
        if total_dice == 0:
            return 0.0
        
        p = 1.0 / self.config["num_faces"]
        expected = total_dice * p
        variance = total_dice * p * (1 - p)
        std_dev = np.sqrt(variance) if variance > 0 else 1.0
        
        # Z-score
        z = (bid.quantity - expected) / std_dev if std_dev > 0 else 0
        
        # Approximate probability (rough estimate)
        prob = max(0.01, min(0.99, 0.5 * (1 + np.tanh(-z))))
        return float(prob)
    
    def encode_trajectory(
        self,
        trajectory: List[Tuple],
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a full game trajectory into a sequence of feature vectors.
        
        Args:
            trajectory: List of (action, player_id, game_state, revealed_dice) tuples
            max_length: Maximum sequence length (pads or truncates)
            
        Returns:
            features: [seq_len, feature_dim] array
            mask: [seq_len] binary mask (1 = real data, 0 = padding)
        """
        if max_length is None:
            max_length = self.max_sequence_length
        
        # Encode each action
        encoded_actions = []
        for item in trajectory:
            if len(item) == 4:
                action, player_id, game_state, revealed_dice = item
            else:
                action, player_id, game_state = item
                revealed_dice = None
            
            features = self.encode_action(action, player_id, game_state, revealed_dice)
            encoded_actions.append(features)
        
        # Convert to array
        if encoded_actions:
            features = np.stack(encoded_actions, axis=0)
        else:
            features = np.zeros((0, self.feature_dim), dtype=np.float32)
        
        # Pad or truncate
        seq_len = features.shape[0]
        
        if seq_len > max_length:
            # Truncate (keep most recent actions)
            features = features[-max_length:]
            mask = np.ones(max_length, dtype=np.float32)
        else:
            # Pad
            mask = np.concatenate([
                np.ones(seq_len, dtype=np.float32),
                np.zeros(max_length - seq_len, dtype=np.float32)
            ])
            padding = np.zeros((max_length - seq_len, self.feature_dim), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        return features, mask
    
    def add_round_markers(
        self,
        trajectory: List[Tuple],
        round_boundaries: List[int]
    ) -> List[Tuple]:
        """
        Add START_ROUND tokens at the beginning of each round.
        
        Args:
            trajectory: List of action tuples
            round_boundaries: List of indices where new rounds start
            
        Returns:
            Modified trajectory with round markers
        """
        # This would need to be implemented if we want explicit round markers
        # For now, the round_ended flag in features serves this purpose
        return trajectory
    
    def batch_encode(
        self,
        trajectories: List[List[Tuple]],
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a batch of trajectories.
        
        Args:
            trajectories: List of trajectory lists
            max_length: Maximum sequence length
            
        Returns:
            features: [batch_size, seq_len, feature_dim]
            masks: [batch_size, seq_len]
        """
        batch_features = []
        batch_masks = []
        
        for traj in trajectories:
            features, mask = self.encode_trajectory(traj, max_length)
            batch_features.append(features)
            batch_masks.append(mask)
        
        return np.stack(batch_features, axis=0), np.stack(batch_masks, axis=0)
    
    def create_special_token_feature(self, token_type: str) -> np.ndarray:
        """
        Create a feature vector for a special token (START_ROUND, END_ROUND, etc.).
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        if token_type == "START_ROUND":
            features[0] = 2.0  # Special value for start round
        elif token_type == "END_ROUND":
            features[0] = 3.0  # Special value for end round
            features[11] = 1.0  # round_ended flag
        elif token_type == "PAD":
            features[0] = -1.0  # Padding indicator
        
        return features
