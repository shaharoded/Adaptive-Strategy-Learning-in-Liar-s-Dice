import os
import random
import pickle
import numpy as np
from collections import defaultdict
from itertools import product, combinations_with_replacement
from torch.utils.tensorboard import SummaryWriter

from . import register_agent
from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid

# --- Approximate Nash/CFR Agent (stub) ---
@register_agent("nash_cfr")
class NashCFRAgent(Agent):
    """
    NashCFRAgent:
    Implements External Sampling Monte Carlo Counterfactual Regret Minimization (MCCFR).
    """
    def __init__(self, policy_dict=None, weights_path=None):
        super().__init__()
        if policy_dict is not None:
            self.policy_dict = policy_dict
        else:
            if weights_path is None:
                weights_path = os.path.join(os.path.dirname(__file__), "weights", "nash_cfr_policy.pkl")
            if not os.path.exists(weights_path):
                raise UntrainedAgentException(f"No trained policy found at {weights_path}.")
            self.policy_dict = NashCFRAgent.load_policy_dict(weights_path)

    def choose_action(self, view):
        my_dice = tuple(sorted(view["my_dice"]))
        last_bid = view["public"].last_bid
        config = view.get("config")
        dice_counts = tuple(view["public"].dice_counts)
        faces = tuple(config.faces)
        total_dice = sum(dice_counts)
        
        # Info set key: (my_dice, last_bid_quantity, last_bid_face)
        if last_bid is None:
            info_set = (my_dice, None, None)
        else:
            info_set = (my_dice, last_bid.quantity, last_bid.face)
            
        policy = None
        key = (dice_counts, faces)
        
        # Try exact match or fallback to single loaded policy
        if self.policy_dict:
            if key in self.policy_dict:
                policy = self.policy_dict[key]
            elif len(self.policy_dict) == 1:
                policy = next(iter(self.policy_dict.values()))
        
        # Sample action from policy
        if policy and info_set in policy:
            action_probs = policy[info_set]
            actions, probs = zip(*action_probs.items())
            chosen = random.choices(actions, weights=probs, k=1)[0]
            if chosen == "call_liar":
                return CallLiarAction()
            else:
                q, f = chosen
                return BidAction(Bid(q, f))
        
        # Fallback: Random legal action
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
                          specific_combinations=None):
        """
        Trains CFR policies for dice count combinations.
        Args:
            specific_combinations: List of tuples (e.g. [(2,2), (2,3)]) to train ONLY specific configs.
        """
        if specific_combinations:
            target_configs = specific_combinations
        else:
            dice_ranges = [range(1, max_dice+1) for _ in range(num_players)]
            target_configs = list(product(*dice_ranges))
        
        policies = {}
        # Load existing checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
                policies = data.get('policies', data) if isinstance(data, dict) else data
                if verbose: print(f"[CFR] Loaded checkpoint with {len(policies)} policies.", flush=True)

        tb_writer = None
        if tensorboard_logdir:
            tb_writer = SummaryWriter(log_dir=tensorboard_logdir)

        total_configs = len(target_configs)
        
        for idx, dice_counts in enumerate(target_configs):
            key = (dice_counts, faces)
            if key in policies and not specific_combinations:
                if verbose: print(f"[CFR] Skipping existing policy for {dice_counts}", flush=True)
                continue
            
            if verbose: print(f"[CFR] [{idx+1}/{total_configs}] Training {dice_counts}...", flush=True)
            
            policy, metrics = NashCFRAgent.train_cfr_policy(
                dice_counts=dice_counts,
                faces=faces,
                iterations=iterations,
                seed=seed,
                track_regret=True
            )
            
            policies[key] = policy
            
            # TensorBoard logging
            if tb_writer:
                d_str = "_".join(map(str, dice_counts))
                if 'convergence_history' in metrics:
                    for i, v in enumerate(metrics['convergence_history']):
                        tb_writer.add_scalar(f"convergence/{d_str}", v, i)
                tb_writer.flush()
            
            # Save checkpoint
            if checkpoint_path:
                save_data = {'policies': policies, 'metrics': {key: metrics}}
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(save_data, f)

        if tb_writer: tb_writer.close()
        return policies

    @staticmethod
    def train_cfr_policy(dice_counts=(2,2), faces=(1,2,3,4,5,6), iterations=10000, seed=42, 
                        track_regret=False, convergence_threshold=0.001, check_convergence_every=500,
                        progress_callback=None):
        """
        External Sampling MCCFR (Monte Carlo Counterfactual Regret Minimization).
        
        Args:
            dice_counts (tuple): Tuple containing number of dice for each player (e.g., (2, 2)).
            faces (tuple): Tuple of valid die faces.
            iterations (int): Number of training iterations.
            seed (int): Random seed for reproducibility.
            track_regret (bool): If True, tracks and checks for strategy convergence.
            convergence_threshold (float): Max allowed delta in strategy probabilities to consider converged.
            check_convergence_every (int): Frequency of convergence checks.
            progress_callback (callable, optional): function(iteration, metric_value) called during training.
            
        Returns:
            tuple: (final_policy_dict, metrics_dict)
        """
        random.seed(seed)
        num_players = len(dice_counts)
        total_dice = sum(dice_counts)
        
        regrets = defaultdict(lambda: defaultdict(float))
        strategy_sum = defaultdict(lambda: defaultdict(float))
        
        # Precompute dice combinations
        dice_combos = [[tuple(sorted(c)) for c in combinations_with_replacement(faces, d)] for d in dice_counts]
        
        def evaluate_terminal(all_dice, last_bid, caller_id):
            """Returns utility for caller: 1 if bid calls a liar correctly, -1 if bid exists."""
            if last_bid is None: return 0
            match_count = sum(d.count(last_bid.face) for d in all_dice)
            # If bid is true (count >= quantity), caller loses (-1). If false, caller wins (1).
            return -1 if match_count >= last_bid.quantity else 1

        def get_strategy(info_set, actions):
            """Computes regret-matching strategy for an info set."""
            strat = {}
            pos_regrets = {a: max(regrets[info_set][a], 0) for a in actions}
            sum_pos = sum(pos_regrets.values())
            if sum_pos > 0:
                for a in actions: strat[a] = pos_regrets[a] / sum_pos
            else:
                for a in actions: strat[a] = 1.0 / len(actions)
            return strat

        def cfr(all_dice, last_bid, current_player, history, traversing_player):
            """Recursive CFR function."""
            # 1. Terminal Check
            if history and history[-1] == "call_liar":
                caller = 1 - current_player # Previous player called
                util = evaluate_terminal(all_dice, last_bid, caller)
                # Utility is relative to traversing_player
                return util if traversing_player == caller else -util

            # 2. Info Set
            my_dice = all_dice[current_player]
            info_set = NashCFRAgent.encode_info_set(my_dice, last_bid, faces)
            actions = NashCFRAgent.legal_actions(last_bid, faces, total_dice)
            strat = get_strategy(info_set, actions)

            # 3. Traversal vs Sampling
            if current_player == traversing_player:
                # TRAVERSAL NODE: Explore ALL actions to update regret
                node_util = 0
                util = {}
                for a in actions:
                    strategy_sum[info_set][a] += strat[a] # Update average strategy
                    
                    if a == "call_liar":
                        util[a] = evaluate_terminal(all_dice, last_bid, current_player)
                    else:
                        next_bid = Bid(a[0], a[1])
                        util[a] = cfr(all_dice, next_bid, 1-current_player, history + [a], traversing_player)
                    node_util += strat[a] * util[a]
                
                # Regret Update
                for a in actions:
                    regrets[info_set][a] += util[a] - node_util
                return node_util
            
            else:
                # OPPONENT NODE: Sample ONE action
                probs = [strat[a] for a in actions]
                chosen = random.choices(actions, weights=probs, k=1)[0]
                
                if chosen == "call_liar":
                    util = evaluate_terminal(all_dice, last_bid, current_player)
                    return -util # traversing_player is NOT caller
                else:
                    next_bid = Bid(chosen[0], chosen[1])
                    return cfr(all_dice, next_bid, 1-current_player, history + [chosen], traversing_player)

        # --- Main Loop ---
        convergence_history = []
        prev_policy = {}
        converged = False
        
        for it in range(iterations):
            # Chance Sampling
            dice_sample = tuple(random.choice(dice_combos[p]) for p in range(num_players))
            
            # Pass 1: Update Player 0
            cfr(dice_sample, None, 0, [], traversing_player=0)
            # Pass 2: Update Player 1
            cfr(dice_sample, None, 0, [], traversing_player=1)

            # Convergence Check (Heuristic)
            if track_regret and (it + 1) % check_convergence_every == 0:
                # 1. Determine keys to check (compare against previous snapshot)
                keys_to_check = list(prev_policy.keys())
                
                if not keys_to_check:
                    # First checkpoint: nothing to compare against
                    max_delta = 1.0 
                else:
                    max_delta = 0
                    for k in keys_to_check:
                        # Re-calculate current strategy for this key
                        acts = strategy_sum[k]
                        total = sum(acts.values())
                        curr_probs = {a: v/total for a, v in acts.items()}
                        
                        # Compare with stored previous strategy
                        prev_probs = prev_policy[k]
                        
                        # Check divergence
                        for a, p in curr_probs.items():
                            diff = abs(p - prev_probs.get(a, 0))
                            if diff > max_delta:
                                max_delta = diff
                        
                        # Also check actions that were in prev but not in current (unlikely with CFR accumulation but possible)
                        for a, p in prev_probs.items():
                            if a not in curr_probs:
                                if p > max_delta:
                                    max_delta = p

                # 2. Prepare snapshot for NEXT checkpoint (Resample to ensure global coverage)
                sample_keys = list(strategy_sum.keys())
                if len(sample_keys) > 100:
                    sample_keys = random.sample(sample_keys, 100)
                
                prev_policy = {}
                for k in sample_keys:
                    acts = strategy_sum[k]
                    total = sum(acts.values())
                    prev_policy[k] = {a: v/total for a, v in acts.items()}

                convergence_history.append(max_delta)
                
                # Output progress via callback
                if progress_callback:
                    progress_callback(it + 1, max_delta)
                    
                if max_delta < convergence_threshold:
                    # print(f"Converged at {it+1}", flush=True) # Worker script handles printing
                    converged = True
                    break

        # Finalize Policy
        final_policy = {}
        for info_set, acts in strategy_sum.items():
            total = sum(acts.values())
            final_policy[info_set] = {a: v/total for a, v in acts.items()} if total > 0 else \
                                     {a: 1.0/len(acts) for a in acts}
        
        metrics = {'convergence_history': convergence_history, 'converged': converged, 'iterations': len(convergence_history) * check_convergence_every}
        return final_policy, metrics

    @staticmethod
    def encode_info_set(my_dice, last_bid, faces):
        if last_bid is None: return (tuple(sorted(my_dice)), None, None)
        return (tuple(sorted(my_dice)), last_bid.quantity, last_bid.face)

    @staticmethod
    def legal_actions(last_bid, faces, total_dice):
        actions = []
        if last_bid is None:
            for f in faces: actions.append((1, f))
        else:
            for q in range(last_bid.quantity, total_dice + 1):
                for f in faces:
                    if q > last_bid.quantity or (q == last_bid.quantity and f > last_bid.face):
                        actions.append((q, f))
            actions.append("call_liar")
        return actions

    @staticmethod
    def save_policy_dict(policy_dict, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: pickle.dump(policy_dict, f)

    @staticmethod
    def load_policy_dict(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data['policies'] if isinstance(data, dict) and 'policies' in data else data