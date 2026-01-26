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
    Uses static methods to train policies for various dice configurations.
    During gameplay, samples actions based on precomputed CFR policies in a stochastic manner.
    Args:
        policy_dict (dict): Pretrained policy dictionary mapping info sets to action probabilities.
                            Allowing to load different policies as long as the policiy is saved in-memory, 
                            avoiding the to load from disk each time.
        weights_path (str): Path to load pretrained policy if policy_dict is not provided.
    
        NOTE: If both policy_dict and weights_path are None, it will attempt to load from default path (weights/nash_cfr_policy.pkl).
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
        
        # Cache for sorted dice and config tuples (reset per round)
        self._cached_my_dice = None
        self._cached_dice_raw = None
        self._cached_config_key = None
        self._cached_config = None

    def choose_action(self, view):
        # Cache sorted dice tuple (only recompute if hand changed)
        my_dice_raw = view["my_dice"]
        if my_dice_raw is not self._cached_dice_raw:
            self._cached_dice_raw = my_dice_raw
            self._cached_my_dice = tuple(sorted(my_dice_raw))
        my_dice = self._cached_my_dice
        
        # Cache config-derived tuples
        config = view.get("config")
        if config is not self._cached_config:
            self._cached_config = config
            self._cached_config_key = (tuple(view["public"].dice_counts), tuple(config.faces))
        
        last_bid = view["public"].last_bid
        dice_counts, faces = self._cached_config_key
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
            
            # Keep sampling until we get a valid action
            max_attempts = len(actions)
            for _ in range(max_attempts):
                chosen = random.choices(actions, weights=probs, k=1)[0]
                if chosen == "call_liar":
                    return CallLiarAction()
                else:
                    q, f = chosen
                    candidate = Bid(q, f)
                    # Check if bid is valid
                    if not self.is_bid_universally_impossible(candidate, total_dice, config):
                        return BidAction(candidate)
            # If all attempts failed, fall through to fallback
        
        # Fallback: Random legal action (should rarely happen if policy is comprehensive)
        if last_bid is None:
            # Try random opening bids until we find a valid one
            for _ in range(len(faces)):
                candidate = Bid(1, random.choice(faces))
                if not self.is_bid_universally_impossible(candidate, total_dice, config):
                    return BidAction(candidate)
            return CallLiarAction()  # Safety fallback if all failed
        # Check if opponent bid is provably false
        if self.is_opponent_bid_provably_false(last_bid, my_dice, total_dice, config):
            return CallLiarAction()
        
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    # Safety guard: skip universally impossible bids
                    if self.is_bid_universally_impossible(candidate, total_dice, config):
                        continue
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
        Each combination is a tuple of dice counts per player (e.g., (2,3) for 2 players with 2 and 3 dice).
        The policy will define strategies for all players in that configuration, based on game's current state.
        Args:
            num_players (int): Number of players in the game.
            max_dice (int): Maximum number of dice per player to train up to.
            faces (tuple): Tuple of valid die faces.
            iterations (int): Number of CFR iterations per configuration, each iteration performs MCCFR.
            seed (int): Random seed for reproducibility.
            verbose (bool): If True, prints progress information.
            checkpoint_path (str): Path to save intermediate policies. If exists, will load and resume.
            tensorboard_logdir (str): Directory to save TensorBoard logs. If None, no logging is done.
            specific_combinations: List of tuples (e.g. [(2,2), (2,3)]) to train ONLY specific configs. Allows resuming, targeted training and parallelization.
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
        Trains a CFR policy for a specific dice configuration.

        MCCFR is a sampling-based variant of Counterfactual Regret Minimization (CFR),
        a popular algorithm for solving extensive-form games. Instead of traversing
        the entire game tree (which can be computationally expensive), MCCFR samples
        game trajectories to estimate regrets and update strategies. This makes it
        scalable to large games like Liar's Dice.

        The training process involves:
        1. Sampling game trajectories based on the current strategy.
        2. Calculating counterfactual regrets for each decision point (info set).
        3. Updating the strategy using regret-matching, which biases future decisions
           toward actions with higher positive regrets.
        4. Repeating the process for a specified number of iterations.

        The goal of MCCFR is to minimize regret over time, leading to a Nash equilibrium
        strategy where no player can unilaterally improve their outcome by deviating.

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
        
        def evaluate_terminal(all_dice, last_bid):
            """
            Returns utility for caller: 1 if bid calls a liar correctly, -1 if bid exists.
            The CFR function will handle perspective switching.
            """
            if last_bid is None: return 0
            match_count = sum(d.count(last_bid.face) for d in all_dice)
            # If bid is true (count >= quantity), caller loses (-1). If false, caller wins (1).
            return -1 if match_count >= last_bid.quantity else 1

        def get_strategy(info_set, actions):
            """
            Computes the regret-matching strategy for a given information set.

            Args:
                info_set (tuple): The current information set (e.g., dice configuration, last bid).
                actions (list): List of legal actions available in the current state.

            Returns:
                dict: A dictionary mapping actions to probabilities based on positive regrets.
            """
            strat = {}
            pos_regrets = {a: max(regrets[info_set][a], 0) for a in actions}
            sum_pos = sum(pos_regrets.values())
            if sum_pos > 0:
                for a in actions: strat[a] = pos_regrets[a] / sum_pos
            else:
                for a in actions: strat[a] = 1.0 / len(actions)
            return strat

        def cfr(all_dice, last_bid, current_player, history, traversing_player):
            """
            Recursive Counterfactual Regret Minimization (CFR) function.

            This function traverses the game tree, updating regrets and strategies for the traversing player.

            Args:
                all_dice (list): List of dice rolls for all players.
                last_bid (Bid or None): The last bid made in the game, or None if no bid has been made.
                current_player (int): The ID of the player whose turn it is.
                history (list): List of actions taken so far in the game.
                traversing_player (int): The ID of the player for whom regrets and strategies are being updated.

            Returns:
                float: The utility of the game state for the traversing player.
            """
            # 1. Terminal Check
            if history and history[-1] == "call_liar":
                caller = 1 - current_player # Previous player called
                util = evaluate_terminal(all_dice, last_bid)
                # Utility is relative to traversing_player
                return util if traversing_player == caller else -util

            # 2. Info Set
            my_dice = all_dice[current_player]
            info_set = NashCFRAgent.encode_info_set(my_dice, last_bid)
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
                        util[a] = evaluate_terminal(all_dice, last_bid)
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
                    util = evaluate_terminal(all_dice, last_bid)
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
    def encode_info_set(my_dice, last_bid):
        """
        Encodes the information set for the current player based on their dice and the last bid.
        """
        if last_bid is None: return (tuple(sorted(my_dice)), None, None)
        return (tuple(sorted(my_dice)), last_bid.quantity, last_bid.face)

    @staticmethod
    def legal_actions(last_bid, faces, total_dice):
        """
        Generates a list of legal actions given the last bid, possible faces, and total dice.
        """
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
        """
        Saves the policy dictionary to the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: pickle.dump(policy_dict, f)

    @staticmethod
    def load_policy_dict(path):
        """
        Loads the policy dictionary from the specified path.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data['policies'] if isinstance(data, dict) and 'policies' in data else data