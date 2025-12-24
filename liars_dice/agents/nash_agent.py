import os
from . import register_agent
from liars_dice.agents.base import Agent, UntrainedAgentException
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid
import random
from itertools import product
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
                          checkpoint_path=None, tensorboard_logdir=None):
        """
        Trains CFR policies for all possible dice count combinations for num_players and up to max_dice per player.
        Returns a dict: (dice_counts, faces) -> policy (info_set -> {action: prob})
        Supports checkpointing and TensorBoard logging.
        """
        dice_ranges = [range(1, max_dice+1) for _ in range(num_players)]
        # Load checkpoint if available
        policies = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                policies = pickle.load(f)
            if verbose:
                print(f"[CFR] Loaded checkpoint with {len(policies)} policies from {checkpoint_path}")
        # TensorBoard setup
        tb_writer = None
        if tensorboard_logdir:
            try:
                tb_writer = SummaryWriter(log_dir=tensorboard_logdir)
            except ImportError:
                print("TensorBoard logging requires torch and tensorboard packages.")
        for dice_counts in product(*dice_ranges):
            key = (dice_counts, faces)
            if key in policies:
                if verbose:
                    print(f"[CFR] Skipping already trained policy for dice_counts={dice_counts}")
                continue
            if verbose:
                print(f"[CFR] Training policy for dice_counts={dice_counts}, faces={faces}...")
            policy, regret_history = NashCFRAgent.train_cfr_policy(
                dice_counts=dice_counts,
                faces=faces,
                iterations=iterations,
                seed=seed,
                track_regret=True,
            )
            policies[key] = policy
            # Log regret to TensorBoard
            if tb_writer:
                for i, avg_regret in enumerate(regret_history):
                    tb_writer.add_scalar(f"regret/dice_{dice_counts}", avg_regret, i)
                tb_writer.flush()
            # Save checkpoint after each policy
            if checkpoint_path:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(policies, f)
        if tb_writer:
            tb_writer.close()
        if verbose:
            print(f"[CFR] Multi-policy training complete. {len(policies)} policies trained.")
        return policies

    @staticmethod
    def save_policy_dict(policy_dict, path=None):
        """
        Save a multi-policy dict to disk using pickle. 
        If path is None, saves to agents/weights/nash_cfr_policy.pkl
        """
        import pickle
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
        """
        import pickle
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "weights", "nash_cfr_policy.pkl")
        if not os.path.exists(path):
            raise UntrainedAgentException(f"No trained policy found at {path}. Please train and save a policy.")
        with open(path, "rb") as f:
            return pickle.load(f)
    
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
    def train_cfr_policy(dice_counts=(2,2), faces=(1,2,3,4,5,6), iterations=10000, seed=42, track_regret=False):
        """
        Trains a CFR policy for Liar's Dice with the given dice_counts (tuple of ints, one per player).
        Returns a policy dict mapping info sets to action probabilities.
        If track_regret=True, also returns a list of average regret per iteration.
        """
        import random
        random.seed(seed)
        from collections import defaultdict
        num_players = len(dice_counts)
        # Regret and strategy tables
        regrets = defaultdict(lambda: defaultdict(float))
        strategy_sum = defaultdict(lambda: defaultdict(float))
        regret_history = []

        def cfr(my_dice, last_bid, player, history, p0, p1):
            info_set = NashCFRAgent.encode_info_set(my_dice, last_bid, faces)
            total_dice = sum(dice_counts)
            actions = NashCFRAgent.legal_actions(last_bid, faces, total_dice)
            # Terminal state: if last action was call liar
            if history and history[-1] == "call_liar":
                # Evaluate terminal utility: +1 if previous bid was true, else -1
                # For demonstration, return 0 utility (stub)
                return 0
            # Get current strategy
            normalizing_sum = 0
            strat = {}
            for a in actions:
                r = regrets[info_set][a]
                strat[a] = max(r, 0)
                normalizing_sum += strat[a]
            if normalizing_sum > 0:
                for a in actions:
                    strat[a] /= normalizing_sum
            else:
                for a in actions:
                    strat[a] = 1.0 / len(actions)
            # Accumulate strategy
            for a in actions:
                strategy_sum[info_set][a] += (p0 if player == 0 else p1) * strat[a]
            # Sample action
            util = {}
            node_util = 0
            for a in actions:
                # For demonstration, just recurse with random dice (not full game tree)
                next_my_dice = my_dice  # Not updating for demonstration
                next_last_bid = last_bid if a == "call_liar" else Bid(a[0], a[1])
                next_history = history + [a]
                if player == 0:
                    util[a] = -cfr(next_my_dice, next_last_bid, 1, next_history, p0 * strat[a], p1)
                else:
                    util[a] = -cfr(next_my_dice, next_last_bid, 0, next_history, p0, p1 * strat[a])
                node_util += strat[a] * util[a]
            # Regret update
            for a in actions:
                regrets[info_set][a] += (util[a] - node_util) * (p1 if player == 0 else p0)
            return node_util

        # Main CFR loop
        for it in range(iterations):
            # For demonstration, use random dice for each player
            dice_samples = [tuple(sorted(random.choices(faces, k=d))) for d in dice_counts]
            for p in range(num_players):
                cfr(dice_samples[p], None, p, [], 1, 1)
            # Track average regret
            if track_regret:
                total_regret = 0
                count = 0
                for info_set in regrets:
                    for a in regrets[info_set]:
                        total_regret += abs(regrets[info_set][a])
                        count += 1
                avg_regret = total_regret / count if count > 0 else 0
                regret_history.append(avg_regret)
            if (it+1) % (iterations//10) == 0:
                print(f"[CFR] Iteration {it+1}/{iterations}")
        # Compute average strategy
        policy = {}
        for info_set, acts in strategy_sum.items():
            total = sum(acts.values())
            if total > 0:
                policy[info_set] = {a: v/total for a, v in acts.items()}
            else:
                n = len(acts)
                policy[info_set] = {a: 1.0/n for a in acts}
        print(f"[CFR] Training complete. Policy has {len(policy)} info sets.")
        if track_regret:
            return policy, regret_history
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
