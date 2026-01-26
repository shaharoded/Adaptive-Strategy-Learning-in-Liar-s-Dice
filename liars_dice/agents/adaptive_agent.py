"""
Adaptive Agent with Neural Opponent Identification

Uses LSTM-based classifier to identify opponent type from action trajectories,
then switches to specialist PPO expert trained against that opponent.
Falls back to generalist PPO (trained by curriculum) when uncertain.
"""

from pathlib import Path
from typing import Optional, Dict

from liars_dice.agents import register_agent
from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.agents.ppo_agent import PPOAgent
from liars_dice.agents.adaptive_agent_utils.adaptive_training import load_neural_classifier
from liars_dice.agents.adaptive_agent_utils.config import CLASSIFIER_CONFIG, PATH_CONFIG
from liars_dice.core.actions import BidAction, CallLiarAction


@register_agent("adaptive")
class AdaptiveAgent(Agent):
    """
    Adaptive agent with LSTM-based opponent identification.
    
    Maintains belief distribution over opponent types and selects specialist
    PPO experts accordingly. Uses manual history synchronization to ensure
    experts have correct context.
    """
    
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        experts_dir: Optional[str] = None,
        generalist_path: Optional[str] = None,
        min_observations: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Args:
            classifier_path: Path to trained neural classifier (default: from PATH_CONFIG)
            experts_dir: Directory containing specialist expert models (default: from PATH_CONFIG)
            generalist_path: Path to generalist PPO model (default: from PATH_CONFIG)
            min_observations: Minimum opponent actions before making prediction (default: from IDENTIFICATION_CONFIG)
            device: torch device ("cpu" or "cuda")
        """
        # Set default paths from config
        if classifier_path is None:
            classifier_path = PATH_CONFIG["neural_classifier"]
        if experts_dir is None:
            experts_dir = PATH_CONFIG["adaptive_models_dir"]
        if generalist_path is None:
            generalist_path = PATH_CONFIG["generalist_model"]
        
        # Resolve paths relative to agents directory
        self.classifier_path = Path(classifier_path)
        if not self.classifier_path.is_absolute():
            self.classifier_path = (Path(__file__).parent / self.classifier_path).resolve()
        
        self.experts_dir = Path(experts_dir)
        if not self.experts_dir.is_absolute():
            self.experts_dir = (Path(__file__).parent / self.experts_dir).resolve()
        
        self.generalist_path = Path(generalist_path)
        if not self.generalist_path.is_absolute():
            self.generalist_path = (Path(__file__).parent / self.generalist_path).resolve()
        
        # Load config defaults if not provided
        self.min_observations = min_observations or CLASSIFIER_CONFIG["min_observations"]
        self.device = device
        
        # Load neural classifier
        try:
            self.belief_tracker = load_neural_classifier(str(self.classifier_path), device=device)
        except FileNotFoundError:
            raise UntrainedAgentException(
                f"Neural classifier not found at {self.classifier_path}. "
                "Train the classifier first using: python scripts/train_adaptive_agent.py --train-classifier"
            )
        
        # Load generalist agent (fallback) - disable auto-sync (use manual sync only)
        try:
            self.generalist = PPOAgent(model_path=str(self.generalist_path), disable_auto_sync=True)
        except Exception as e:
            raise UntrainedAgentException(
                f"Generalist PPO agent not found at {self.generalist_path}: {e}"
            )
        
        # Load all specialist experts
        self.experts: Dict[str, PPOAgent] = {}
        self._load_experts()
        
        # State tracking
        self.current_expert: Optional[str] = None
        self.observations_count = 0
        self.round_index = -1
        self.last_public_state = None
        self.round_action_history = []  # Track actions in current round for expert sync
        
        print(f"AdaptiveAgent initialized (Neural LSTM):")
        print(f"  Min observations: {self.min_observations}")
        print(f"  Device: {self.device}")
        print(f"  Available experts: {len(self.experts)}")
        print(f"  Opponent types: {self.belief_tracker.opponent_types}")
    
    def _load_experts(self):
        """Load all specialist expert models."""
        for opp_type in self.belief_tracker.opponent_types:
            safe_name = opp_type.replace(" ", "_").replace("/", "_")
            expert_path = self.experts_dir / f"expert_{safe_name}"
            
            try:
                # Load with disable_auto_sync=True so we can manually manage history
                expert = PPOAgent(model_path=str(expert_path), disable_auto_sync=True)
                self.experts[opp_type] = expert
                print(f"  [OK] Loaded expert for {opp_type}")
            except Exception as e:
                print(f"  [WARNING] Could not load expert for {opp_type}: {e}")
                # Continue without this expert - will use generalist as fallback
    
    def reset(self):
        """Reset agent state for a new game."""
        self.belief_tracker.reset()
        self.current_expert = None
        self.observations_count = 0
        self.round_index = -1
        self.last_public_state = None
        self.round_action_history = []
        
        # Reset all expert agents (clear their history buffers)
        self.generalist.reset()
        for expert in self.experts.values():
            expert.reset()
        for agent in [self.generalist, *self.experts.values()]:
            self._clear_agent_buffers(agent, -1)
    
    def choose_action(self, view):
        """
        Chooses an action using adaptive expert selection based on neural belief tracking.
        
        This method manages the transition between the generalist agent and specialized
        experts. It ensures that whenever an expert is called—especially mid-round or 
        when playing as Player 1 (P1)—the expert's internal history buffer and bid 
        tracking are perfectly synchronized with the observed game state.
        
        Args:
            view (dict): The current game view provided by the engine.
            
        Returns:
            Action: The BidAction or CallLiarAction selected by the active agent.
        """
        public = view["public"]
        
        # 1. Detect Round/Game Transitions
        if self.round_index != -1 and public.round_index == 0 and self.round_index != 0:
            # Full Match Reset
            self.reset()
        if public.round_index != self.round_index:
            # New round detected from our perspective
            self._handle_round_transition(public.round_index)
            
        # Store state for sync
        self.last_public_state = public

        # 2. Select Agent
        selected_agent = self._select_agent()
        
        # 3. Critical Sync: Rebuild the expert's view of the world
        self._sync_history_to_agent(selected_agent, public)
        
        # Force the expert to match the current table bid
        if hasattr(selected_agent, 'set_last_bid'):
            selected_agent.set_last_bid(public.last_bid)
        
        # 4. Action
        action = selected_agent.choose_action(view)
        
        # 5. Record My Action
        self.round_action_history.append({
            'is_me': True,
            'action': action,
            'is_bid': isinstance(action, BidAction),
            'bid': action.bid if isinstance(action, BidAction) else None
        })
        
        # If I called liar, round is over. Clear history so Round N+1 starts clean.
        if isinstance(action, CallLiarAction):
            self.round_action_history = []
        
        return action

    def _clear_agent_buffers(self, agent: Agent, round_index: int):
        """Hard reset of an agent's internal buffers for a given round."""
        if hasattr(agent, 'history_buffer'):
            agent.history_buffer = []
        if hasattr(agent, 'encoder') and hasattr(agent.encoder, 'history_buffer'):
            agent.encoder.history_buffer = []
        if hasattr(agent, 'last_bid_on_table'):
            agent.last_bid_on_table = None
        if hasattr(agent, 'last_round_idx'):
            agent.last_round_idx = round_index

    def _handle_round_transition(self, new_round_index: int):
        """Clear per-round buffers when a new round is observed."""
        self.round_index = new_round_index
        self.round_action_history = []
        for agent in [self.generalist, *self.experts.values()]:
            self._clear_agent_buffers(agent, new_round_index)
    
    def record_opponent_action(self, action, game_state: Dict, revealed_dice=None):
        """
        Directly record an opponent action that was observed in order to update beliefs.
        
        Args:
            action: The opponent's action (BidAction or CallLiarAction)
            game_state: Game state context
            revealed_dice: Revealed dice if round ended
        """
        # Detect round transitions early (P1 opener comes here before our choose_action)
        incoming_round = game_state.get("round_index", self.round_index)
        if self.round_index != -1 and incoming_round == 0 and self.round_index != 0:
            self.reset()
        if incoming_round != self.round_index:
            self._handle_round_transition(incoming_round)

        # Add to trajectory
        self.belief_tracker.update_belief(
            action,
            player_id=1,  # Opponent
            game_state=game_state,
            revealed_dice=revealed_dice
        )
        self.observations_count += 1
        
        # Record in round history for expert sync
        self.round_action_history.append({
            'is_me': False,
            'action': action,
            'is_bid': isinstance(action, BidAction),
            'bid': action.bid if isinstance(action, BidAction) else None
        })
        
        # If opponent called liar, round ended - clear history for next round
        if isinstance(action, CallLiarAction):
            self.round_action_history = []
    
    def _sync_history_to_agent(self, agent: Agent, public):
        """
        Forcefully overwrites the agent's history buffer AND its encoder's internal state.
        This solves the P1 Blindness by pushing the opponent's first bid into the expert.
        
        Args:
            agent: The PPO agent to sync history to
            public: Public game state (for bid tracking)
        """
        if not hasattr(agent, 'history_buffer'):
            return
        if not hasattr(agent, 'encoder') or agent.encoder is None:
            return
        
        max_bid_qty = agent.encoder.max_bid_qty
        synced_history = []
        for action_record in self.round_action_history:
            is_me = 1.0 if action_record['is_me'] else 0.0
            is_bid = 1.0 if action_record['is_bid'] else 0.0
            bid = action_record['bid']
            qty = bid.quantity / max_bid_qty if bid else 0.0
            face = bid.face / 6.0 if bid else 0.0
            synced_history.append([is_me, is_bid, qty, face])
        synced_copy = [list(v) for v in synced_history]

        # Overwrite both locations to ensure no stale data remains
        agent.history_buffer = synced_copy
        if hasattr(agent, 'encoder'):
            # This forces the HistoryObservationEncoder to use our NEW data for its next obs
            agent.encoder.history_buffer = [list(v) for v in synced_copy]
        if hasattr(agent, 'last_round_idx'):
            agent.last_round_idx = self.round_index
        if hasattr(agent, 'last_bid_on_table'):
            agent.last_bid_on_table = public.last_bid

    
    def _select_agent(self) -> Agent:
        """
        Select which agent to use based on current beliefs.
        
        Strategy:
        - If observations < min_observations: use generalist (not enough data)
        - If observations >= min_observations: use argmax opponent immediately
        
        At every step, we reassess which expert to use based on updated beliefs.
        This ensures smooth, responsive expert switching.
        
        Returns:
            The selected PPO agent (expert or generalist)
        """
        # Need minimum observations before making a decision
        if self.observations_count < self.min_observations:
            return self.generalist
        
        # After minimum observations, use argmax to select the most likely opponent
        beliefs = self.belief_tracker.get_belief_distribution()
        best_opponent = max(beliefs.items(), key=lambda x: x[1])[0]
        
        # Switch expert if prediction changed
        if best_opponent != self.current_expert:
            # Check if expert exists for this opponent
            if best_opponent in self.experts:
                self.current_expert = best_opponent
            else:
                # Expert not available - use generalist
                self.current_expert = None
        
        # Return selected agent
        if self.current_expert is not None and self.current_expert in self.experts:
            return self.experts[self.current_expert]
        else:
            return self.generalist
    
    def get_belief_summary(self) -> Dict:
        """
        Get summary of current belief state for monitoring/debugging.
        
        Returns:
            Dict with belief distribution, entropy, selected expert, etc.
        """
        beliefs = self.belief_tracker.get_belief_distribution()
        entropy = self.belief_tracker.get_entropy()
        
        return {
            "beliefs": beliefs,
            "entropy": entropy,
            "observations": self.observations_count,
            "current_expert": self.current_expert,
            "using_generalist": self.current_expert is None
        }
