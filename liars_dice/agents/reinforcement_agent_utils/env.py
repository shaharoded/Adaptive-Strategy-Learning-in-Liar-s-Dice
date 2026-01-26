import gymnasium as gym
import numpy as np
from gymnasium import spaces

from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.config import GameConfig
from liars_dice.core.bid import Bid
from liars_dice.core.actions import CallLiarAction, BidAction
from liars_dice.core.reward import get_reward
from liars_dice.agents.reinforcement_agent_utils.encoder import HistoryObservationEncoder


class LiarsDiceGymEnv(gym.Env):
    """
    A Gymnasium environment for Liar's Dice (1v1) compatible with RL libraries.
    
    Logic Flow:
    -----------
    1. RL Agent calls step(action).
    2. Env applies action.
    3. Env simulates Opponent turns until:
       a) It is the RL Agent's turn again.
       b) The round ends (Terminal state).
    
    Observation Space:
    ------------------
    A vector containing:
    - The Agent's private hand.
    - Public game counts (dice remaining).
    - A history buffer of the last N moves in the current round (The Trajectory).
    
    Action Space:
    -------------
    Discrete(N). We flatten (Quantity, Face) into a single integer index.
    Masking is provided via 'get_action_mask' to prevent illegal moves.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, game_config: GameConfig, opponent_agent_cls, randomize_position=True):
        """
        Args:
            game_config: Configuration object (dice count, etc).
            opponent_agent_cls: The class of the opponent (e.g. RandomAgent, CFRAgent).
                                This agent is instantiated once and reset per round.
            randomize_position: If True, randomly assigns RL agent to player 0 or 1 each episode.
                                This helps the agent learn from both first and second player perspectives.
        """
        super().__init__()
        self.cfg = game_config
        self.randomize_position = randomize_position
        
        # Initialize the Encoder with History
        self.encoder = HistoryObservationEncoder(game_config, history_len=10)
        
        # --- Internal Game State ---
        self.engine = GameEngine(self.cfg)
        self.opponent_agent_cls = opponent_agent_cls  # Store class, not instance
        self.opponent = None  # Will be recreated periodically in reset()
        self.opponent_batch_size = 5  # Keep same opponent for 5 episodes, then resample
        self.episodes_with_current_opponent = 0  # Track episode count with current opponent
        self.rl_player_id = 0  # Will be randomized in reset() if randomize_position=True
        
        # --- Full Match State (dice elimination) ---
        self.initial_dice_per_player = game_config.total_dice  # total_dice is per-player count
        self.rl_agent_dice = self.initial_dice_per_player
        self.opponent_dice = self.initial_dice_per_player
        self.match_winner = None  # Track winner for full match
        
        # --- Action Space Setup ---
        # 0 = Call Liar
        # 1..N = Bids (Flattened)
        # Max bid = total dice in game (num_players * dice_per_player)
        self.max_bid_quantity = self.cfg.total_dice * self.cfg.num_players 
        self.num_faces = 6
        self.n_bids = self.max_bid_quantity * self.num_faces
        self.action_space = spaces.Discrete(1 + self.n_bids)

        # --- Observation Space Setup ---
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.encoder.shape, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new FULL MATCH (not just a round).
        1. Resets dice counts for both players to initial values.
        2. Clears engine state.
        3. Clears history buffer.
        4. Randomizes RL agent position (if enabled).
        5. If Opponent goes first, simulates until RL Agent's turn.
        """
        super().reset(seed=seed)
        
        # 1. Recreate opponent periodically (batched, not every episode)
        # This allows agent to learn from opponent patterns while maintaining diversity
        should_recreate = (
            self.opponent is None or 
            self.episodes_with_current_opponent >= self.opponent_batch_size
        )
        
        if should_recreate:
            self.opponent = self.opponent_agent_cls()
            self.episodes_with_current_opponent = 0
            
            # Special handling for LeagueOpponent - resample opponent
            if hasattr(self.opponent, 'reset_for_new_episode'):
                self.opponent.reset_for_new_episode()
        else:
            # Keep same opponent but resample if it has reset capability
            if hasattr(self.opponent, 'reset_for_new_episode'):
                self.opponent.reset_for_new_episode()
        
        self.episodes_with_current_opponent += 1
        
        # 2. Randomize RL agent position (helps learn from both perspectives)
        if self.randomize_position:
            self.rl_player_id = np.random.randint(0, 2)  # Randomly 0 or 1
        else:
            self.rl_player_id = 0
        
        # 3. Reset Match State (dice counts)
        self.rl_agent_dice = self.initial_dice_per_player
        self.opponent_dice = self.initial_dice_per_player
        self.match_winner = None
        
        # 4. Reset Engine
        self.engine = GameEngine(self.cfg) 
        self._start_new_round_with_dice_counts()
        
        # 5. Reset Encoder History
        self.encoder.reset()
        
        # 6. Handle Opponent Turn (if they are Player 0)
        if self.engine.state.public.current_player != self.rl_player_id:
            self._play_opponent_turn_sequence()
            
        return self._get_obs(), self._get_info()
    
    def _start_new_round_with_dice_counts(self):
        """Start a new round with current dice counts from match state."""
        # Update player dice counts based on match state
        # Respect which player is the RL agent
        if self.rl_player_id == 0:
            self.engine.state.players[0].num_dice = self.rl_agent_dice
            self.engine.state.players[1].num_dice = self.opponent_dice
        else:  # rl_player_id == 1
            self.engine.state.players[0].num_dice = self.opponent_dice
            self.engine.state.players[1].num_dice = self.rl_agent_dice
        self.engine.start_new_round()

    def get_action_mask(self):
        """
        Public method required by MaskablePPO to get the action mask.
        Returns the current action mask as a numpy boolean array.
        """
        return self._get_action_mask()

    def step(self, action_idx):
        """
        Run one timestep of the environment's dynamics.
        
        This Step Function handles the "Meta-Turn":
        1. Agent plays Action.
        2. If game continues, Opponent plays their response(s).
        3. Returns control when it is Agent's turn again (or game over).
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Decode Action (Index -> Object)
        action = self._decode_action(action_idx)
        
        try:
            # Update History for Encoder BEFORE applying (so we record our own move)
            self.encoder.add_event(self.rl_player_id, action, self.rl_player_id)
            
            # Apply Action to Engine
            self.engine.apply_action(self.rl_player_id, action)
            
            # Collect Immediate Rewards (e.g., if I called Liar)
            # We pop events generated by OUR move
            total_reward += self._process_events_and_rewards(action)
            
        except IllegalMoveError:
            # Safety Net: Ideally MaskablePPO prevents this. 
            # If it happens, we end the match with a penalty.
            total_reward = -10.0 
            terminated = True
            info = self._get_info()
            info['error'] = "IllegalMove"
            return self._get_obs(), total_reward, terminated, truncated, info

        # 2. If round is not over, Play Opponent Sequence
        if not self.engine.is_terminal():
            opp_reward = self._play_opponent_turn_sequence()
            total_reward += opp_reward

        # 3. Handle Round End - Dice Elimination
        if self.engine.is_terminal():
            round_loser = self.engine.state.public.loser
            
            # Update match state: loser loses a die
            if round_loser == self.rl_player_id:
                self.rl_agent_dice -= 1
            else:
                self.opponent_dice -= 1
            
            # Check if match is over (someone has 0 dice)
            if self.rl_agent_dice == 0:
                self.match_winner = 1  # Opponent wins
                total_reward += -10.0  # Large penalty for losing the match
                terminated = True
            elif self.opponent_dice == 0:
                self.match_winner = 0  # RL agent wins
                total_reward += 10.0  # Large reward for winning the match
                terminated = True
            else:
                # Match continues - start a new round with reduced dice
                self.encoder.reset()  # Clear history for new round
                self._start_new_round_with_dice_counts()
                
                # If opponent goes first in the new round, simulate their turn
                if self.engine.state.public.current_player != self.rl_player_id:
                    opp_reward = self._play_opponent_turn_sequence()
                    total_reward += opp_reward
        
        # 4. Final termination check
        terminated = (self.match_winner is not None)
        
        info = self._get_info()
        if terminated and self.match_winner is not None:
            info['match_winner'] = self.match_winner
        
        return self._get_obs(), total_reward, terminated, truncated, info

    def _play_opponent_turn_sequence(self):
        """
        Internal loop: Let the opponent play until it is the RL agent's turn again.
        Returns:
            float: Sum of rewards accumulated during opponent's turns.
        """
        accumulated_reward = 0.0
        
        while (not self.engine.is_terminal()) and (self.engine.state.public.current_player != self.rl_player_id):
            curr_player = self.engine.state.public.current_player
            view = self.engine.get_view(curr_player)
            
            # Opponent Logic
            opp_action = self.opponent.choose_action(view)
            
            # Update History (Record Opponent Move)
            self.encoder.add_event(curr_player, opp_action, self.rl_player_id)
            
            # Apply Move
            self.engine.apply_action(curr_player, opp_action)
            
            # Calculate reward (e.g. if opponent bluffed and lost, I might get +1)
            # We pass the opponent's action to calculate rewards relative to ME (rl_player_id)
            accumulated_reward += self._process_events_and_rewards(opp_action)
            
        return accumulated_reward

    def _process_events_and_rewards(self, last_action):
        """
        Helper: Pops events from engine queue, calculates rewards using user's `get_reward`.
        """
        step_reward = 0.0
        events = self.engine.pop_events()
        
        # We need the current view to calculate rewards (context)
        view = self.engine.get_view(self.rl_player_id)
        
        for ev in events:
            # We specifically ask: "What is the reward for the RL Player (id=0) given this event?"
            r = get_reward(
                event_type=ev.get('type'),
                state=view, 
                action=last_action, 
                player=self.rl_player_id, # Always calculate reward for RL agent
                public_state=self.engine.state.public,
                event_details=ev  # Pass full event for extra info like was_true
            )
            step_reward += r
        return step_reward

    def _get_obs(self):
        """Delegates observation creation to the HistoryEncoder."""
        view = self.engine.get_view(self.rl_player_id)
        
        # Extract hand from view dict - my_dice is a tuple of dice values
        my_dice = view["my_dice"]
        hand_dict = {}
        for die in my_dice:
            hand_dict[die] = hand_dict.get(die, 0) + 1
        
        # Get dice counts
        my_dice_count = len(my_dice)
        
        # Calculate opponent dice count from public state
        public = view["public"]
        opp_dice_count = sum(c for i, c in enumerate(public.dice_counts) if i != self.rl_player_id)
        
        return self.encoder.encode(hand_dict, my_dice_count, opp_dice_count)

    def _get_info(self):
        """Returns the Action Mask required by MaskablePPO."""
        return {"action_mask": self._get_action_mask()}

    def _get_action_mask(self):
        """
        Generates a boolean mask [True, False, ...] where True indicates a valid move.
        This prevents the RL agent from learning invalid bids.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # 1. 'Call Liar' Validity
        if self.engine.state.public.last_bid is not None:
            mask[0] = True # Valid if a bid exists
        else:
            mask[0] = False # Invalid on very first turn
        
        # 2. Bidding Validity
        current_bid = self.engine.state.public.last_bid
        
        # Iterate over all possible bids in action space
        for q in range(1, self.max_bid_quantity + 1):
            for f in range(1, 7):
                idx = 1 + (q-1)*6 + (f-1)
                
                if idx < self.action_space.n:
                    potential_bid = Bid(quantity=q, face=f)
                    
                    # Logic: Must be strictly greater than current bid
                    if current_bid is None:
                        mask[idx] = True
                    elif potential_bid.is_higher_than(current_bid):
                        mask[idx] = True
                    else:
                        mask[idx] = False
        return mask

    def _decode_action(self, action_idx):
        """Maps discrete integer index -> GameAction object."""
        if action_idx == 0:
            return CallLiarAction()
        
        # Reverse mapping: idx = 1 + (q-1)*6 + (f-1)
        adjusted = action_idx - 1
        quantity = (adjusted // 6) + 1
        face = (adjusted % 6) + 1
        
        return BidAction(Bid(quantity, face))