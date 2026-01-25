"""
adaptive_agent.py

Neural LSTM-based Adaptive Agent that identifies opponents and selects specialist experts.

This agent:
1. Maintains a neural belief tracker that uses LSTM to process action trajectories
2. Updates beliefs at each step based on observed opponent actions
3. Once confident about opponent identity, switches to the specialist expert
4. Falls back to generalist PPO agent if uncertain
"""

from pathlib import Path
from typing import Optional, Dict

from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.agents.ppo_agent import PPOAgent
from liars_dice.agents import register_agent
from liars_dice.agents.adapter_agent.adaptive_training import load_neural_classifier
from liars_dice.agents.adapter_agent.config import IDENTIFICATION_CONFIG, PATH_CONFIG


@register_agent("adaptive")
class AdaptiveAgent(Agent):
    """
    Adaptive agent that uses neural LSTM to identify opponents
    and selects specialist PPO experts accordingly.
    
    The agent uses an LSTM-based classifier to process action trajectories
    and predict opponent type. Once confident, it switches to the specialist
    expert trained against that opponent type.
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
        self.min_observations = min_observations or IDENTIFICATION_CONFIG["min_observations"]
        self.device = device
        
        # Load neural classifier
        try:
            self.belief_tracker = load_neural_classifier(str(self.classifier_path), device=device)
        except FileNotFoundError:
            raise UntrainedAgentException(
                f"Neural classifier not found at {self.classifier_path}. "
                "Train the classifier first using: python scripts/train_adaptive_agent.py --train-classifier"
            )
        
        # Load generalist agent (fallback)
        try:
            self.generalist = PPOAgent(model_path=str(self.generalist_path))
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
                expert = PPOAgent(model_path=str(expert_path))
                self.experts[opp_type] = expert
                print(f"  ✓ Loaded expert for {opp_type}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load expert for {opp_type}: {e}")
                # Continue without this expert - will use generalist as fallback
    
    def reset(self):
        """Reset agent state for a new game."""
        self.belief_tracker.reset()
        self.current_expert = None
        self.observations_count = 0
        self.round_index = -1
        self.last_public_state = None
    
    def choose_action(self, view):
        """
        Choose action using adaptive expert selection with neural belief tracking.
        
        Process:
        1. Receive opponent actions via ActionTrackerWrapper's record_opponent_action() calls
        2. Update neural belief tracker with opponent's trajectory
        3. Select expert based on current beliefs (argmax after min_observations)
        4. Delegate action choice to selected expert (or generalist)        
        """
        # Extract state information
        public = view["public"]
        
        # Check for new round (reset if needed)
        if public.round_index != self.round_index:
            if self.round_index != -1 and public.round_index == 0:
                # New game - reset beliefs
                self.reset()
            self.round_index = public.round_index
        
        # Select expert based on current beliefs (opponent actions recorded via wrapper)
        selected_agent = self._select_agent()
        
        # Delegate action to selected agent
        action = selected_agent.choose_action(view)
        
        return action
    
    def record_opponent_action(self, action, game_state: Dict, revealed_dice=None):
        """
        Directly record an opponent action that was observed.
        This is the proper way to update beliefs - not by inferring from state changes.
        
        Args:
            action: The opponent's action (BidAction or CallLiarAction)
            game_state: Game state context
            revealed_dice: Revealed dice if round ended
        """
        # Add to trajectory
        self.belief_tracker.update_belief(
            action,
            player_id=1,  # Opponent
            game_state=game_state,
            revealed_dice=revealed_dice
        )
        self.observations_count += 1
    
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
