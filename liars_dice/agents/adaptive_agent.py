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

from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid

from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.agents.ppo_agent import PPOAgent
from liars_dice.agents import register_agent
from liars_dice.agents.adapter_agent.neural_belief_tracker import NeuralBeliefTracker
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
        confidence_threshold: Optional[float] = None,
        min_observations: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Args:
            classifier_path: Path to trained neural classifier (default: from PATH_CONFIG)
            experts_dir: Directory containing specialist expert models (default: from PATH_CONFIG)
            generalist_path: Path to generalist PPO model (default: from PATH_CONFIG)
            confidence_threshold: Minimum belief probability to select expert (default: from IDENTIFICATION_CONFIG)
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
        self.confidence_threshold = confidence_threshold or IDENTIFICATION_CONFIG["confidence_threshold"]
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
        print(f"  Confidence threshold: {self.confidence_threshold:.1%}")
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
        self.last_opponent_action = None
        self.last_game_state = None
        self.round_index = -1
    
    def choose_action(self, view):
        """
        Choose action using adaptive expert selection with neural belief tracking.
        
        Process:
        1. Detect opponent actions from state changes
        2. Update neural belief tracker with opponent's trajectory
        3. Select expert based on current beliefs
        4. Delegate action choice to selected expert (or generalist)
        """
        # Extract state information
        public = view["public"]
        my_dice = view.get("my_dice", [])
        player_id = view.get("player_id", 0)
        
        # Check for new round (reset if needed)
        if public.round_index != self.round_index:
            if self.round_index != -1 and public.round_index == 0:
                # New game - reset beliefs
                self.reset()
            self.round_index = public.round_index
        
        # Build game state context
        game_state = {
            "last_bid": public.last_bid,
            "total_dice": sum(public.dice_counts),
            "my_dice_count": len(my_dice),
            "opp_dice_count": sum(c for i, c in enumerate(public.dice_counts) if i != player_id),
            "round_index": public.round_index
        }
        
        # Detect opponent action from state change
        if self.last_public_state is not None:
            opponent_action = self._infer_opponent_action(self.last_public_state, public)
            
            if opponent_action is not None:
                # Update neural belief tracker
                revealed_dice = None  # We don't have access to revealed dice in this context
                self.belief_tracker.update_belief(
                    opponent_action,
                    player_id=1,  # Opponent
                    game_state=game_state,
                    revealed_dice=revealed_dice
                )
                self.observations_count += 1
                
                # Log belief update periodically
                if self.observations_count % 5 == 0:
                    beliefs = self.belief_tracker.get_belief_distribution()
                    entropy = self.belief_tracker.get_entropy()
                    top_belief = max(beliefs.items(), key=lambda x: x[1])
                    print(f"[Adaptive] Observations: {self.observations_count}, "
                          f"Entropy: {entropy:.3f}, Top: {top_belief[0]} ({top_belief[1]:.1%})")
        
        # Store current state for next comparison
        self.last_public_state = public
        
        # Select expert based on beliefs
        selected_agent = self._select_agent()
        
        # Delegate action to selected agent
        action = selected_agent.choose_action(view)
        
        return action
    
    def _infer_opponent_action(self, prev_public, curr_public):
        """
        Infer opponent action from public state transition.
        """       
        prev_bid = prev_public.last_bid
        curr_bid = curr_public.last_bid
        
        # Check if bid changed
        if prev_bid != curr_bid:
            if curr_bid is not None:
                # Opponent made a bid
                return BidAction(Bid(curr_bid.quantity, curr_bid.face))
            else:
                # Round ended (opponent called liar)
                return CallLiarAction()
        
        return None
    
    def _select_agent(self) -> Agent:
        """
        Select which agent to use based on current beliefs.
        
        Returns:
            The selected PPO agent (expert or generalist)
        """
        # Need minimum observations before making a decision
        if self.observations_count < self.min_observations:
            return self.generalist
        
        # Get most likely opponent
        predicted_opponent = self.belief_tracker.get_best_opponent(
            confidence_threshold=self.confidence_threshold
        )
        
        # Check if we should switch expert
        if predicted_opponent != self.current_expert:
            if predicted_opponent is not None:
                # Switch to specialist expert
                if predicted_opponent in self.experts:
                    self.current_expert = predicted_opponent
                    beliefs = self.belief_tracker.get_belief_distribution()
                    print(f"\\n[Adaptive] Switching to expert for {predicted_opponent} "
                          f"(confidence: {beliefs[predicted_opponent]:.1%})\\n")
                else:
                    # Expert not available - use generalist
                    print(f"\\n[Adaptive] Expert for {predicted_opponent} not available, "
                          f"using generalist\\n")
                    return self.generalist
            else:
                # Not confident - use generalist
                self.current_expert = None
        
        # Return selected agent
        if self.current_expert and self.current_expert in self.experts:
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
            "confidence_threshold": self.confidence_threshold,
            "using_generalist": self.current_expert is None
        }
