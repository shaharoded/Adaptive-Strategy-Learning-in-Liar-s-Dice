"""
ppo_config.py
Central configuration for PPO Agent architecture and training hyperparameters.
"""

MODEL_CONFIG = {
    "policy_type": "MlpPolicy",
    "learning_rate": 3e-4,
    "gamma": 0.99,          # Discount factor (high because only terminal reward matters)
    "batch_size": 128,      # Increased from 64 to reduce variance in gradients
    "ent_coef": 0.05,       # Reduced from 0.1 for curriculum (higher in league if needed)
    "history_length": 10,   # Size of the sliding window for memory
    "policy_kwargs": {
        "net_arch": [256, 256] # Custom architecture: 2 hidden layers of 256 neurons
    }
}

TRAINING_CONFIG = {
    "total_timesteps": 100_000, # Total training timesteps (steps taken by the agent, keep high as the agent will auto-stop if converged)
    "log_dir": "./runs/ppo_training/",
    "model_save_path": "./liars_dice/agents/weights/ppo_model",
}