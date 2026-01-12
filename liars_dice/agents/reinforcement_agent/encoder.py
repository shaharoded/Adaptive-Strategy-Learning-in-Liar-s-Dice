import numpy as np


class HistoryObservationEncoder:
    """
    Shared Logic for converting game data into a flat vector for the Neural Network.
    Used by both the Training Env (God View) and the Inference Agent (Player View).
    Encodes the Game State + The Trajectory (History) of the current round as a single vecctor.
    This architecture choice—a Flattened Sliding Window (or "Time-Delay Embedding")-is a standard method to
    enable simple Feed-Forward Networks (MLPs) to process temporal data 
    without the complexity of Recurrent Neural Networks (RNNs).
    
    The observation vector is composed of:
    1. Static Features: My Hand, Total Dice, Opponent Dice Count.
    2. Sequence History: The last N actions taken in the round (by anyone).
    
    Each action in history is encoded as a vector of 4 values:
    [Is_My_Action, Action_Type, Bid_Quantity, Bid_Face]
    
    If History Length is 10, and each action is 4 floats, the history part is 40 floats.
    """
    def __init__(self, game_config=None, total_dice=None, history_len=10):
        # Support both game_config object and direct total_dice parameter
        if game_config is not None:
            self.max_dice = game_config.total_dice if hasattr(game_config, 'total_dice') else 5
        elif total_dice is not None:
            self.max_dice = total_dice
        else:
            self.max_dice = 5  # Default fallback
            
        self.history_len = history_len
        
        # --- Normalization Constants ---
        self.max_bid_qty = self.max_dice + 5 
        
        # --- Dimensions ---
        # 1. My Hand: 6 faces
        # 2. Game Context: 3 floats
        # 3. History: history_len * 4 features
        self.obs_dim = 6 + 3 + (self.history_len * 4)
        
        # Buffer to store the raw history of the current round
        self.history_buffer = []

    def reset(self):
        """Clears the history buffer for a new round."""
        self.history_buffer = []

    def add_event(self, player_id, action, agent_id):
        """Adds an action to the history buffer.
        
        Args:
            player_id: ID of the player who took the action
            action: The action object (BidAction or CallLiarAction)
            agent_id: ID of the RL agent (to determine if this is "my" action)
        """
        is_me = 1.0 if player_id == agent_id else 0.0
        
        action_type = type(action).__name__
        if action_type == "BidAction":
            is_bid = 1.0
            qty = action.bid.quantity / self.max_bid_qty
            face = action.bid.face / 6.0
        else: # CallLiarAction
            is_bid = 0.0
            qty = 0.0
            face = 0.0
            
        features = [is_me, is_bid, qty, face]
        self.history_buffer.append(features)

    def encode(self, hand_dict, my_dice_count, opp_dice_count):
        """
        Generates the full numpy observation vector.
        Accepts generic counts to support both Training (God view) and Inference (Player view).
        
        Args:
            hand_dict: Dictionary mapping face (1-6) -> count
            my_dice_count: Number of dice I have
            opp_dice_count: Number of dice opponent has
        """
        # 1. My Hand (6 floats)
        hand_vec = np.zeros(6, dtype=np.float32)
        for f, count in hand_dict.items():
            if 1 <= f <= 6:
                hand_vec[f-1] = count / self.max_dice
        
        # 2. Context (3 floats)
        total_dice = (my_dice_count + opp_dice_count)
        context_vec = np.array([
            my_dice_count / self.max_dice,
            opp_dice_count / self.max_dice, 
            total_dice / self.max_dice
        ], dtype=np.float32)
        
        # 3. History (history_len * 4 floats)
        hist_vec = np.zeros((self.history_len, 4), dtype=np.float32)
        
        if self.history_buffer:
            recent = self.history_buffer[-self.history_len:]
            start_idx = self.history_len - len(recent)
            hist_vec[start_idx:, :] = np.array(recent, dtype=np.float32)
            
        hist_flat = hist_vec.flatten()
        
        return np.concatenate([hand_vec, context_vec, hist_flat]).astype(np.float32)

    @property
    def shape(self):
        return (self.obs_dim,)