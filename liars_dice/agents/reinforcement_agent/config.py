"""
ppo_config.py
Central configuration for PPO Agent architecture and training hyperparameters.
"""

MODEL_CONFIG = {
    "policy_type": "MlpPolicy",
    "learning_rate": 3e-4,
    "gamma": 0.99,          # Discount factor (high because only terminal reward matters)
    "batch_size": 64,
    "ent_coef": 0.01,       # Entropy coefficient to encourage exploration
    "policy_kwargs": {
        "net_arch": [256, 256] # Custom architecture: 2 hidden layers of 256 neurons
    }
}

TRAINING_CONFIG = {
    "total_timesteps": 100_000,
    "save_freq": 10_000,
    "log_dir": "./runs/ppo_training/",
    "model_save_path": "./liars_dice/agents/weights/ppo_model",
    "history_length": 10,   # Size of the sliding window for memory
}