"""
Adaptive Agent Package

Contains the neural LSTM-based adaptive agent that identifies opponents and selects specialist experts.
"""

from liars_dice.agents.adapter_agent.neural_belief_tracker import NeuralBeliefTracker, OpponentClassifierLSTM
from liars_dice.agents.adapter_agent.trajectory_encoder import TrajectoryEncoder
from liars_dice.agents.adapter_agent.adaptive_training import (
    train_specialist_expert,
    train_neural_classifier,
    collect_opponent_trajectories,
    evaluate_specialist_experts,
    load_neural_classifier
)

__all__ = [
    "NeuralBeliefTracker",
    "OpponentClassifierLSTM",
    "TrajectoryEncoder",
    "train_specialist_expert",
    "train_neural_classifier",
    "collect_opponent_trajectories",
    "evaluate_specialist_experts",
    "load_neural_classifier"
]
