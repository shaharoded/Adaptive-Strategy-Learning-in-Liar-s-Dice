"""
reward.py
Defines reward calculation logic for Liar's Dice ML/RL data collection.
Allows easy modification of reward schemes for different experiments.
"""

def get_reward(event_type, state, action, player, public_state=None, event_details=None):
    """
    Returns the reward for a given event, state, and action.
    Includes reward shaping to guide intermediate decisions, not just terminal rewards.
    
    Args:
        event_type (str): Type of event (e.g., 'BidPlaced', 'LiarCalled', 'RoundEnded', 'Error').
        state (dict): Player's view/state at this step.
        action (object): Action taken.
        player (int): Player index.
        public_state (object|dict): Public state (for winner/loser info).
        event_details (dict): Full event dict with extra info like was_true.
    Returns:
        float: Reward value.
    """
    if event_type == "Error":
        return -1.0
    
    # Reward for calling liar when it's TRUE (good decision)
    if event_type == "RoundEnded" and public_state is not None:
        winner = getattr(public_state, "winner", None)
        was_true = event_details.get('was_true') if event_details else None
        
        if winner is not None:
            if player == winner:
                # Winner of the round - stronger reward for calling correct liar
                if was_true is not None:
                    if was_true:
                        # Called liar when bid WAS true - agent loses (should have bid instead)
                        return -0.5  
                    else:
                        # Called liar when bid was FALSE - good decision!
                        return 2.0
                else:
                    # Regular win (from opponent calling liar on your bid that was true)
                    return 1.0
            else:
                # Loser of the round
                if was_true is not None:
                    if was_true:
                        # Your bid was true but opponent called liar - your mistake was not betting higher
                        return -1.5
                    else:
                        # Your bid was false and opponent called liar - bad decision
                        return -2.0
                else:
                    # Regular loss
                    return -1.0
        return 0.0
    
    return 0.0
