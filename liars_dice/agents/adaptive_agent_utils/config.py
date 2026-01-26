"""
config.py

Configuration for Adaptive Agent architecture and training.
Defines the neural network architecture for opponent identification
and training hyperparameters.
"""

# ==============================================================================
# RUNTIME CONFIGURATION (used during gameplay)
# ==============================================================================

# Path Configuration
PATH_CONFIG = {
    # Models
    "weights_dir": "weights",
    "adaptive_models_dir": "weights/adaptive_models",
    "neural_classifier": "weights/adaptive_models/neural_classifier.pt",
    "expert_prefix": "weights/adaptive_models/expert",
    "generalist_model": "weights/ppo_model",

    # Logs
    "expert_log_dir": "./runs/adaptive_agent/expert_training",
    "classifier_log_dir": "./runs/adaptive_agent/classifier",
}

# ==============================================================================
# TRAINING CONFIGURATION (used during classifier training only)
# ==============================================================================

# Classifier Training
TRAINING_CONFIG = {
    # Data collection
    "samples_per_opponent": 1000,    # Games to collect per opponent
    "exclude_random_agents": True,   # Exclude purely random agents (they lack learnable patterns)
    
    # Progressive sampling: Generate samples at different trajectory lengths
    # This teaches the classifier to identify opponents from SHORT trajectories
    "progressive_sampling": True,    # Enable progressive sampling
    
    # Agents to exclude if exclude_random_agents=True
    "excluded_agent_patterns": [
        "RandomAgent",
        "CautiousRandomAgent", 
        "AggressiveRandomAgent",
        "FaceFixedRandomAgent",
        "FaceRandomRandomAgent",
        # Note: We keep RandomFaceAgent and RandomThresholdAgent (they have heuristic components)
    ],
    
    # Neural network training hyperparameters
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 50,
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "min_loss_improvement": 1e-4,   # Minimum loss decrease to consider as improvement
    "train_val_split": 0.8,         # Train/validation split ratio
}

# Neural Classifier Architecture
CLASSIFIER_CONFIG = {
    # Network architecture
    "hidden_dim": 128,              # LSTM hidden dimension
    "num_lstm_layers": 2,           # Number of LSTM layers
    "dropout": 0.2,                 # Dropout for regularization
    "embedding_dim": 64,            # Embedding dimension for categorical features
    
    # Input sequence
    "max_sequence_length": 50,      # Maximum trajectory length to consider
    "action_vocab_size": 100,       # Vocabulary size for action encoding

    "min_observations": 5,          # Min opponent actions before making prediction
    "update_frequency": 1,          # How often to update beliefs (every N actions)
}

# Expert PPO Training (override default PPO settings as needed)
EXPERT_CONFIG = {
    "timesteps": 500_000,           # Training timesteps per expert
    "win_rate_threshold": 0.99,     # Target 99% win rate before stopping (higher than regular PPO)
    "enable_early_stopping": True,
}

# Trajectory Encoder Configuration
ENCODER_CONFIG = {
    # Special tokens
    "special_tokens": {
        "PAD": 0,                   # Padding token
        "START_ROUND": 1,           # Start of new round
        "END_ROUND": 2,             # End of round (reveal)
        "CALL_LIAR": 3,             # Call liar action
    },
    
    # Feature dimensions
    "num_faces": 6,                 # Dice faces (1-6)
    "max_quantity": 10,             # Maximum bid quantity (2 players, 5 dice max)
    "max_dice_per_player": 5,       # Maximum dice per player
    
    # Encoding strategy
    "encode_revealed_dice": True,   # Include revealed dice in encoding
    "encode_relative_bid": True,    # Encode relative bid increase
    "encode_probability": True,     # Include bid probability estimates
}
