"""
action_tracker.py

Wraps opponent agents to track their actions and pass them to the adaptive agent's
belief tracker. This ensures the LSTM classifier receives all opponent moves.
"""

from liars_dice.agents.base import Agent
from liars_dice.agents.adaptive_agent import AdaptiveAgent


class ActionTrackerWrapper:
    """
    Wraps an opponent agent and records all actions it takes.
    
    This enables the adaptive agent to collect opponent trajectories by:
    1. Intercepting calls to opponent.choose_action()
    2. Capturing the returned action
    3. Forwarding it to the adaptive agent's belief tracker
    """
    
    def __init__(self, opponent_agent: Agent, adaptive_agent: 'AdaptiveAgent' = None):
        """
        Args:
            opponent_agent: The opponent to wrap
            adaptive_agent: The adaptive agent whose belief tracker should be updated
        """
        self.opponent = opponent_agent
        self.adaptive_agent = adaptive_agent
        self.last_view = None
    
    def choose_action(self, view):
        """
        Choose action from wrapped opponent and record it.
        
        Args:
            view: Game view (provided by engine)
            
        Returns:
            The opponent's action
        """
        # Get action from opponent
        action = self.opponent.choose_action(view)
        
        # Record action in adaptive agent if provided
        if self.adaptive_agent is not None:
            public = view["public"]
            my_dice = view.get("my_dice", [])
            player_id = view.get("player_id", 0)
            
            # Build game state
            game_state = {
                "last_bid": public.last_bid,
                "total_dice": sum(public.dice_counts),
                "my_dice_count": len(my_dice),
                "opp_dice_count": sum(c for i, c in enumerate(public.dice_counts) if i != player_id),
                "round_index": public.round_index
            }
            
            # Record opponent action (always player 1 in 2-player game where adaptive agent perspective varies)
            self.adaptive_agent.record_opponent_action(action, game_state, revealed_dice=None)
        
        return action
    
    def __getattr__(self, name):
        """Delegate all other method calls to the wrapped opponent."""
        return getattr(self.opponent, name)
