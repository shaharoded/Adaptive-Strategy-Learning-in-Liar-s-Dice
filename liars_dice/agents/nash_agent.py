import os
from . import register_agent
from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid
import random
import numpy as np
from collections import defaultdict
from itertools import product, combinations_with_replacement
import pickle
from torch.utils.tensorboard import SummaryWriter


# --- Approximate Nash/CFR Agent (stub) ---
@register_agent("nash_cfr")
class NashCFRAgent(Agent):
    """
    NashCFRAgent:
    - Implements a Counterfactual Regret Minimization (CFR) agent for Liar's Dice.
    - Uses a precomputed policy table mapping information sets (private dice, public state, last bid) to action probabilities.
    - If no policy is loaded, acts randomly as fallback.

    CFR Training:
    - Use the static method train_cfr_policy to generate a policy table for a given game configuration.
    - This method runs self-play CFR for a specified number of iterations and returns a policy dict.
    - The policy can be saved/loaded as needed and passed to the agent at initialization.
    """
    def __init__(self, policy_dict=None, weights_path=None):
        """
        Args:
            policy_dict: Optional dict mapping (dice_counts, faces) -> policy (info_set -> {action: prob})
                dice_counts: tuple of ints, one per player (e.g., (2,1))
                If None, will attempt to load from weights_path.
            weights_path: Optional path to policy pickle file. If None, defaults to agents/weights/nash_cfr_policy.pkl
        """
        super().__init__()
        if policy_dict is not None:
            self.policy_dict = policy_dict
        else:
            if weights_path is None:
                weights_path = os.path.join(os.path.dirname(__file__), "weights", "nash_cfr_policy.pkl")
            if not os.path.exists(weights_path):
                raise UntrainedAgentException(f"No trained policy found at {weights_path}. Please train and save a policy.")
            self.policy_dict = NashCFRAgent.load_policy_dict(weights_path)
        self.weights_path = weights_path if weights_path is not None else os.path.join(os.path.dirname(__file__), "weights", "nash_cfr_policy.pkl")

    def choose_action(self, view):
        my_dice = tuple(sorted(view["my_dice"]))
        last_bid = view["public"].last_bid
        config = view.get("config")
        dice_counts = tuple(view["public"].dice_counts)
        faces = tuple(config.faces)
        total_dice = sum(dice_counts)
        # Build info set key: (my_dice, last_bid_quantity, last_bid_face)
        if last_bid is None:
            info_set = (my_dice, None, None)
        else:
            info_set = (my_dice, last_bid.quantity, last_bid.face)
        # Policy selection: try exact match, else fallback to any available
        policy = None
        key = (dice_counts, faces)
        if self.policy_dict:
            if key in self.policy_dict:
                policy = self.policy_dict[key]
            elif len(self.policy_dict) == 1:
                # Single-policy dict, use for all configs
                policy = next(iter(self.policy_dict.values()))
        # If policy is loaded, sample action from policy
        if policy and info_set in policy:
            action_probs = policy[info_set]
            actions, probs = zip(*action_probs.items())
            chosen = random.choices(actions, weights=probs, k=1)[0]
            if chosen == "call_liar":
                return CallLiarAction()
            else:
                q, f = chosen
                return BidAction(Bid(q, f))
        # Fallback: random legal action
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()
    
    @staticmethod
    def train_multi_policy(num_players=2, max_dice=5, faces=(1, 2, 3, 4, 5, 6),
                          iterations=10000, seed=42, verbose=True,
                          checkpoint_path=None, tensorboard_logdir=None,
                          convergence_threshold=0.001, adaptive_training=True):
        """
        Trains CFR policies for all possible dice count combinations for num_players and up to max_dice per player.
        
        Enhanced features:
        - Automatic convergence detection per policy
        - Adaptive training for under-explored states
        - Comprehensive TensorBoard logging:
          * Regret curves per dice configuration
          * Convergence metrics
          * State coverage
          * Training efficiency metrics
        
        Returns a dict: (dice_counts, faces) -> policy (info_set -> {action: prob})
        Supports checkpointing and TensorBoard logging.
        """
        dice_ranges = [range(1, max_dice+1) for _ in range(num_players)]
        # Load checkpoint if available
        policies = {}
        training_metrics = {}  # Store metrics for each config
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                if isinstance(checkpoint_data, dict):
                    # Check if it's the new format with metrics
                    if 'policies' in checkpoint_data:
                        policies = checkpoint_data['policies']
                        training_metrics = checkpoint_data.get('metrics', {})
                    else:
                        # Old format, just policies
                        policies = checkpoint_data
            if verbose:
                print(f"[CFR] Loaded checkpoint with {len(policies)} policies from {checkpoint_path}")
        
        # TensorBoard setup
        tb_writer = None
        if tensorboard_logdir:
            try:
                tb_writer = SummaryWriter(log_dir=tensorboard_logdir)
                if verbose:
                    print(f"[CFR] TensorBoard logging to {tensorboard_logdir}")
            except ImportError:
                print("[CFR] TensorBoard logging requires torch and tensorboard packages.")
        
        total_configs = sum(1 for _ in product(*dice_ranges))
        config_idx = 0
        
        for dice_counts in product(*dice_ranges):
            config_idx += 1
            key = (dice_counts, faces)
            
            if key in policies:
                if verbose:
                    print(f"[CFR] [{config_idx}/{total_configs}] Skipping already trained policy for dice_counts={dice_counts}")
                continue
            
            if verbose:
                print(f"[CFR] [{config_idx}/{total_configs}] Training policy for dice_counts={dice_counts}, faces={faces}...")
            
            policy, metrics = NashCFRAgent.train_cfr_policy(
                dice_counts=dice_counts,
                faces=faces,
                iterations=iterations,
                seed=seed,
                track_regret=True,
                convergence_threshold=convergence_threshold,
                adaptive_training=adaptive_training,
            )
            
            policies[key] = policy
            training_metrics[key] = metrics
            
            # Enhanced TensorBoard logging
            if tb_writer:
                dice_str = "_".join(map(str, dice_counts))
                
                # Normalized regret (this is the one you want for “is it stabilizing?”)
                for i, r_norm in enumerate(metrics.get('regret_norm_history', [])):
                    tb_writer.add_scalar(f"regret_norm/dice_{dice_str}", r_norm, i)

                # Regret delta (should trend toward ~0)
                for i, r_delta in enumerate(metrics.get('regret_delta_history', [])):
                    tb_writer.add_scalar(f"regret_delta/dice_{dice_str}", r_delta, i)
                
                # Log convergence curve
                for i, conv in enumerate(metrics['convergence_history']):
                    tb_writer.add_scalar(f"convergence/dice_{dice_str}", conv, i)
                
                # Log state coverage growth
                for i, coverage in enumerate(metrics['state_coverage_history']):
                    tb_writer.add_scalar(f"coverage/dice_{dice_str}", coverage, i)
                
                # Log summary statistics
                tb_writer.add_scalar(f"summary/iterations_dice_{dice_str}", 
                                   metrics['actual_iterations'], config_idx)
                tb_writer.add_scalar(f"summary/converged_dice_{dice_str}", 
                                   1 if metrics['converged'] else 0, config_idx)
                tb_writer.add_scalar(f"summary/info_sets_dice_{dice_str}", 
                                   metrics['final_info_sets'], config_idx)
                tb_writer.add_scalar(f"summary/states_visited_dice_{dice_str}", 
                                   len(metrics['state_visits']), config_idx)
                
                # Log state visit distribution
                if metrics['state_visits']:
                    visit_counts = list(metrics['state_visits'].values())
                    tb_writer.add_histogram(f"state_visits/dice_{dice_str}", 
                                          np.array(visit_counts), config_idx)
                    avg_visits = sum(visit_counts) / len(visit_counts)
                    min_visits = min(visit_counts)
                    max_visits = max(visit_counts)
                    tb_writer.add_scalar(f"visits/avg_dice_{dice_str}", avg_visits, config_idx)
                    tb_writer.add_scalar(f"visits/min_dice_{dice_str}", min_visits, config_idx)
                    tb_writer.add_scalar(f"visits/max_dice_{dice_str}", max_visits, config_idx)
                
                tb_writer.flush()
            
            # Save checkpoint after each policy (with metrics)
            if checkpoint_path:
                checkpoint_data = {
                    'policies': policies,
                    'metrics': training_metrics
                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)
                if verbose:
                    print(f"[CFR] Checkpoint saved to {checkpoint_path}")
        
        if tb_writer:
            # Log overall training summary
            tb_writer.add_text("training_summary", 
                             f"Total configs: {total_configs}\n"
                             f"Policies trained: {len(policies)}\n"
                             f"Iterations per config: {iterations}\n"
                             f"Convergence threshold: {convergence_threshold}\n"
                             f"Adaptive training: {adaptive_training}")
            tb_writer.close()
        
        if verbose:
            print(f"[CFR] Multi-policy training complete. {len(policies)} policies trained.")
            converged_count = sum(1 for m in training_metrics.values() if m['converged'])
            print(f"[CFR] Converged policies: {converged_count}/{len(policies)}")
        
        return policies

    @staticmethod
    def save_policy_dict(policy_dict, path=None):
        """
        Save a multi-policy dict to disk using pickle. 
        If path is None, saves to agents/weights/nash_cfr_policy.pkl
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "weights", "nash_cfr_policy.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(policy_dict, f)

    @staticmethod
    def load_policy_dict(path=None):
        """
        Load a multi-policy dict from disk using pickle. 
        If path is None, loads from agents/weights/nash_cfr_policy.pkl
        Handles both old format (just policies) and new format (policies + metrics).
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "weights", "nash_cfr_policy.pkl")
        if not os.path.exists(path):
            raise UntrainedAgentException(f"No trained policy found at {path}. Please train and save a policy.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Handle new format with metrics
        if isinstance(data, dict) and 'policies' in data:
            return data['policies']
        # Old format, just policies
        return data
    
    @staticmethod
    def encode_info_set(my_dice, last_bid, faces):
        """Encodes the information set as a tuple for policy/regret lookup."""
        if last_bid is None:
            return (tuple(sorted(my_dice)), None, None)
        return (tuple(sorted(my_dice)), last_bid.quantity, last_bid.face)

    @staticmethod
    def legal_actions(last_bid, faces, total_dice):
        """Returns all legal actions (bid tuples or 'call_liar') from the current state."""
        actions = []
        if last_bid is None:
            for f in faces:
                actions.append((1, f))
        else:
            for q in range(last_bid.quantity, total_dice + 1):
                for f in faces:
                    candidate = Bid(q, f)
                    if candidate.is_higher_than(last_bid):
                        actions.append((q, f))
            actions.append("call_liar")
        return actions

    @staticmethod
    def train_cfr_policy(dice_counts=(2,2), faces=(1,2,3,4,5,6), iterations=10000, seed=42, 
                        track_regret=False, convergence_threshold=0.001, check_convergence_every=100,
                        min_iterations=1000, adaptive_training=True, exploration_bonus=0.1):
        """
        Trains a CFR policy for Liar's Dice with the given dice_counts (tuple of ints, one per player).
        
        Enhanced features:
        - Proper game tree traversal with all dice combinations
        - Convergence detection (auto-stop when strategy stabilizes)
        - State visitation tracking to identify under-explored areas
        - Adaptive training: allocate more iterations to under-explored states
        
        Args:
            dice_counts: Tuple of dice per player (e.g., (2,2))
            faces: Tuple of possible dice faces
            iterations: Maximum iterations (may stop early on convergence)
            seed: Random seed
            track_regret: Return regret history
            convergence_threshold: Strategy change threshold for convergence
            check_convergence_every: Check convergence every N iterations
            min_iterations: Minimum iterations before checking convergence
            adaptive_training: Enable adaptive iteration allocation
            exploration_bonus: Bonus weight for under-explored states
            
        Returns:
            policy dict or (policy, metrics) if track_regret=True
        """
        random.seed(seed)
        num_players = len(dice_counts)
        total_dice = sum(dice_counts)
        
        # Regret and strategy tables
        regrets = defaultdict(lambda: defaultdict(float))
        strategy_sum = defaultdict(lambda: defaultdict(float))
        state_visits = defaultdict(int)  # Track state visitation
        
        # Metrics tracking
        regret_history = []
        convergence_history = []
        state_coverage_history = []
        regret_norm_history = []
        regret_delta_history = []
        
        # Generate all possible dice combinations for each player
        def generate_all_dice_combinations(num_dice, faces):
            """Generate all possible dice outcomes for a player."""
            return [tuple(sorted(combo)) for combo in combinations_with_replacement(faces, num_dice)]
        
        dice_combinations = [generate_all_dice_combinations(d, faces) for d in dice_counts]
        
        def evaluate_terminal(all_dice, last_bid, caller_id):
            """
            Evaluate terminal utility when liar is called.
            Returns utility from perspective of player 0.
            """
            if last_bid is None:
                return 0  # No bid to challenge
            
            # Count how many dice match the bid face
            match_count = sum(dice.count(last_bid.face) for dice in all_dice)
            bid_is_true = match_count >= last_bid.quantity
            
            # Caller wins if bid was false, loses if bid was true
            if bid_is_true:
                # Bid was true, caller loses
                utility = 1 if caller_id == 1 else -1
            else:
                # Bid was false, caller wins
                utility = -1 if caller_id == 1 else 1
            
            return utility
        
        def get_strategy(info_set, actions, regrets_dict):
            """Get current strategy using regret matching."""
            normalizing_sum = 0
            strat = {}
            for a in actions:
                r = regrets_dict[info_set][a]
                strat[a] = max(r, 0)
                normalizing_sum += strat[a]
            
            if normalizing_sum > 0:
                for a in actions:
                    strat[a] /= normalizing_sum
            else:
                # Uniform strategy
                for a in actions:
                    strat[a] = 1.0 / len(actions)
            
            return strat
        
        def cfr(all_dice, last_bid, player, history, p0, p1, depth=0, max_depth=20):
            """
            CFR with full game tree traversal.
            all_dice: tuple of tuples, dice for each player
            """
            # Depth limit to prevent infinite recursion
            if depth > max_depth:
                return 0
            
            my_dice = all_dice[player]
            info_set = NashCFRAgent.encode_info_set(my_dice, last_bid, faces)
            state_visits[info_set] += 1
            
            actions = NashCFRAgent.legal_actions(last_bid, faces, total_dice)
            
            # Terminal state: if last action was call_liar
            if history and history[-1] == "call_liar":
                return evaluate_terminal(all_dice, last_bid, player)
            
            # Get current strategy
            strat = get_strategy(info_set, actions, regrets)
            
            # Accumulate strategy
            reach_prob = p0 if player == 0 else p1
            for a in actions:
                strategy_sum[info_set][a] += reach_prob * strat[a]
            
            # Compute utilities for each action
            util = {}
            node_util = 0
            
            for a in actions:
                if a == "call_liar":
                    # Terminal: evaluate immediately
                    util[a] = evaluate_terminal(all_dice, last_bid, player)
                else:
                    # Recurse to next player
                    next_last_bid = Bid(a[0], a[1])
                    next_player = 1 - player
                    next_history = history + [a]
                    
                    if player == 0:
                        util[a] = -cfr(all_dice, next_last_bid, next_player, next_history, 
                                      p0 * strat[a], p1, depth + 1, max_depth)
                    else:
                        util[a] = -cfr(all_dice, next_last_bid, next_player, next_history,
                                      p0, p1 * strat[a], depth + 1, max_depth)
                
                node_util += strat[a] * util[a]
            
            # Regret update
            counterfactual_prob = p1 if player == 0 else p0
            for a in actions:
                regrets[info_set][a] += counterfactual_prob * (util[a] - node_util)
            
            return node_util
        
        # Previous policy for convergence check
        prev_policy = {}
        converged = False
        actual_iterations = 0
        
        print(f"[CFR] Starting training for dice_counts={dice_counts}")
        print(f"[CFR] Total dice combinations: {[len(dc) for dc in dice_combinations]}")
        
        # Main CFR loop
        for it in range(iterations):
            actual_iterations = it + 1
            
            # Sample dice combinations - explore all combinations over time
            if adaptive_training and it > min_iterations:
                # Adaptive sampling: favor less-visited states
                # Weight by inverse visitation count
                weights = []
                all_combos = []
                for combo in zip(*[dice_combinations[p] for p in range(num_players)]):
                    all_combos.append(combo)
                    # Compute average visit count for this dice configuration
                    avg_visits = sum(state_visits.get(NashCFRAgent.encode_info_set(combo[p], None, faces), 0) 
                                   for p in range(num_players)) / num_players
                    weight = 1.0 / (avg_visits + exploration_bonus)
                    weights.append(weight)
                
                # Sample proportional to inverse visits
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                dice_sample = random.choices(all_combos, weights=probs, k=1)[0]
            else:
                # Random uniform sampling
                dice_sample = tuple(random.choice(dice_combinations[p]) for p in range(num_players))
            
            # Run CFR for each player
            for p in range(num_players):
                cfr(dice_sample, None, p, [], 1, 1)
            
            # Track metrics
            if track_regret:
                # Average absolute regret
                total_regret = sum(abs(r) for info_set in regrets for r in regrets[info_set].values())
                count = sum(len(regrets[info_set]) for info_set in regrets)
                avg_regret = total_regret / count if count > 0 else 0
                regret_history.append(avg_regret)

                # Normalized regret: avg regret per training step
                regret_norm = avg_regret / (it + 1)
                regret_norm_history.append(regret_norm)

                # Per-iteration change (helps visually flatten)
                if len(regret_history) >= 2:
                    regret_delta_history.append(regret_history[-1] - regret_history[-2])
                else:
                    regret_delta_history.append(0.0)
                
                # State coverage
                state_coverage = len(state_visits)
                state_coverage_history.append(state_coverage)
            
            # Convergence check
            if it >= min_iterations and (it + 1) % check_convergence_every == 0:
                # Compute current policy
                current_policy = {}
                for info_set, acts in strategy_sum.items():
                    total = sum(acts.values())
                    if total > 0:
                        current_policy[info_set] = {a: v/total for a, v in acts.items()}
                    else:
                        n = len(acts)
                        current_policy[info_set] = {a: 1.0/n for a in acts}
                
                # Compare with previous policy
                if prev_policy:
                    max_change = 0
                    for info_set in current_policy:
                        if info_set in prev_policy:
                            for action in current_policy[info_set]:
                                change = abs(current_policy[info_set].get(action, 0) - 
                                           prev_policy[info_set].get(action, 0))
                                max_change = max(max_change, change)
                    
                    convergence_history.append(max_change)
                    
                    if max_change < convergence_threshold:
                        converged = True
                        print(f"[CFR] Converged at iteration {it+1} (max change: {max_change:.6f})")
                        break
                
                prev_policy = current_policy
            
            # Progress reporting
            if (it + 1) % max(1, iterations // 10) == 0:
                coverage = len(state_visits)
                avg_visits = sum(state_visits.values()) / len(state_visits) if state_visits else 0
                print(f"[CFR] Iteration {it+1}/{iterations} | Info sets: {len(regrets)} | "
                      f"States visited: {coverage} | Avg visits: {avg_visits:.1f}")
        
        # Compute final average strategy
        policy = {}
        for info_set, acts in strategy_sum.items():
            total = sum(acts.values())
            if total > 0:
                policy[info_set] = {a: v/total for a, v in acts.items()}
            else:
                n = len(acts)
                policy[info_set] = {a: 1.0/n for a in acts}
        
        print(f"[CFR] Training complete. Policy has {len(policy)} info sets.")
        print(f"[CFR] Actual iterations: {actual_iterations} | Converged: {converged}")
        print(f"[CFR] Unique states visited: {len(state_visits)}")
        
        if track_regret:
            metrics = {
                'regret_history': regret_history,
                'convergence_history': convergence_history,
                'state_coverage_history': state_coverage_history,
                'regret_norm_history': regret_norm_history,
                'regret_delta_history': regret_delta_history,
                'state_visits': dict(state_visits),
                'actual_iterations': actual_iterations,
                'converged': converged,
                'final_info_sets': len(policy)
            }
            return policy, metrics
        return policy


"""
USAGE EXAMPLES:
----------------
Train and save multi-policy dict for 2 players, dice counts 1-5 (default location):

    policies = NashCFRAgent.train_multi_policy(num_players=2, max_dice=5, faces=(1,2,3,4,5,6), iterations=10000)
    NashCFRAgent.save_policy_dict(policies)  # saves to agents/weights/nash_cfr_policy.pkl

Load and use in agent (default):

    agent = NashCFRAgent()  # loads from agents/weights/nash_cfr_policy.pkl

If the policy file is missing, UntrainedAgent will be raised.
You can also specify a custom path:

    agent = NashCFRAgent(weights_path="/path/to/policy.pkl")
"""
