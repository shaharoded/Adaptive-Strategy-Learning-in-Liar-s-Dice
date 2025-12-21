"""
reward.py
Defines reward calculation logic for Liar's Dice ML/RL data collection.
Allows easy modification of reward schemes for different experiments.
"""

def get_reward(event_type, state, action, player, public_state=None):
    """
    Returns the reward for a given event, state, and action.
    Default scheme:
      - 0 for intermediate steps
      - +1 for winner at RoundEnded, -1 for loser
      - -1 for error events
    Args:
        event_type (str): Type of event (e.g., 'BidPlaced', 'RoundEnded', 'Error').
        state (dict): Player's view/state at this step.
        action (object): Action taken.
        player (int): Player index.
        public_state (object|dict): Public state (for winner/loser info).
    Returns:
        int: Reward value.
    """
    if event_type == "Error":
        return -1
    if event_type == "RoundEnded" and public_state is not None:
        winner = getattr(public_state, "winner", None)
        if winner is not None:
            if player == winner:
                return 1
            else:
                return -1
        return 0
    return 0
