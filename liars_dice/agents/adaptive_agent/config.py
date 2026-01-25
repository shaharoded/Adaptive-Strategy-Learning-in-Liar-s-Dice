"""
config.py

Configuration for Adaptive Agent architecture and training.
Defines the neural network architecture for opponent identification
and training hyperparameters.
"""

# Neural Classifier Architecture
CLASSIFIER_CONFIG = {
    # Network architecture
    "hidden_dim": 128,              # LSTM hidden dimension
    "num_lstm_layers": 2,           # Number of LSTM layers
    "dropout": 0.2,                 # Dropout for regularization
    "embedding_dim": 64,            # Embedding dimension for categorical features
    
    # Input features
    "max_sequence_length": 50,      # Maximum trajectory length to consider
    "action_vocab_size": 100,       # Vocabulary size for action encoding
    
    # Training
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 50,
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "min_loss_improvement": 1e-4,  # Minimum loss decrease to consider as improvement
    
    # Data collection
    "train_val_split": 0.8,         # Train/validation split ratio
}

# Expert Training Configuration
EXPERT_CONFIG = {
    "timesteps": 500_000,           # Training timesteps per expert
    "win_rate_threshold": 0.99,     # Target 99% win rate
    "enable_early_stopping": True,
    "log_dir": "./runs/adaptive_agent",
}

# Path Configuration
PATH_CONFIG = {
    "weights_dir": "weights",
    "adaptive_models_dir": "weights/adaptive_models",
    "neural_classifier": "weights/adaptive_models/neural_classifier.pt",
    "expert_prefix": "weights/adaptive_models/expert",
    "generalist_model": "weights/ppo_model",
    "tensorboard_log_dir": "./runs/adaptive_agent/classifier",
}

# Opponent Identification Thresholds
IDENTIFICATION_CONFIG = {
    "min_observations": 4,          # Min actions before making prediction
    "update_frequency": 1,          # How often to update beliefs (every N actions)
}

# Encoder Configuration
ENCODER_CONFIG = {
    # Token types
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
