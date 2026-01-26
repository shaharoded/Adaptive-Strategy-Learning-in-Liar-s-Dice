"""
An early version of the adaptive agent,
This agent and it's training process manages an external memory of best "hats" (agents) in a specific state of the game.
This is a strategy optimizer Mixture-of-Experts implementation, designed to pick "the best horse" at a given state.
"""


from liars_dice.agents import register_agent
from .base import Agent, UntrainedAgentException
from liars_dice.agents.hat_adapter_agent_utils.memory import Memory
from liars_dice.agents.random_agent import RandomAgent
from liars_dice.agents.bayesian_agent import BayesianAgent
from liars_dice.agents.heuristic_agent import ConservativeAgent
from liars_dice.agents.nash_agent import NashCFRAgent

from collections import Counter
from typing import Dict, Any


@register_agent("hat_adapter_agent")
class HatAdapterAgent(Agent):
    """
    An adaptive agent that selects among a set of expert agents ("hats") based on learned performance in different game states.
    It uses a memory of state-action performance to choose the most suitable expert for the current game situation.

    NOTE: This agent does not require bid validation checks, as it's next action is always produced by a valid expert agent.
    """
    def __init__(self, memory_file='memory_checkpoint_7000000_games.pkl'):
        super().__init__()
        self.memory = Memory(memory_file)
        self.memory.load()
        self.agents = self._load_agents()
        self.fallback_agent = RandomAgent()

        self._cache = {}
        self._cache_max = 50_000
        self._score_cache = {}

    def _load_agents(self):
        agents: Dict[str, Any] = {
            "RandomAgent": RandomAgent(),
            "BayesianAgent": BayesianAgent(),
            "HeuristicAgent": ConservativeAgent(),
        }
        try:
            agents["NashAgent"] = NashCFRAgent()
        except UntrainedAgentException:
            agents["NashAgent"] = BayesianAgent()
        return agents

    def choose_action(self, state_view):
        # Two-stage selection for better behavior:
        # 1) Ask each expert what it would do.
        # 2) Score the expert using the table that matches that action type.
        # This prevents the meta-agent from ignoring last_bid/turn because an expert that bids crazy-high
        # tends to get punished in Bid-table for early-game states.

        state_representation = self._get_state_representation(state_view)

        public = state_view["public"]
        opening = public.last_bid is None

        proposals = []  # (agent_name, expert, action_type, action)
        for name, expert in self.agents.items():
            act = expert.choose_action(state_view)
            action_type = act.__class__.__name__
            # normalize to our memory table keys
            if action_type == "BidAction":
                tkey = "Bid"
            elif action_type == "CallLiarAction":
                tkey = "CallLiar"
            else:
                tkey = None
            proposals.append((name, expert, tkey, act))

        # If opening, only bids are legal, so just pick best Bid-proposer
        if opening:
            proposals = [p for p in proposals if p[2] == "Bid"]
            if not proposals:
                return self.fallback_agent.choose_action(state_view)

        # Score each proposal using the right table first, and fall back to global table.
        best = None
        best_score = None
        for name, expert, tkey, act in proposals:
            cache_key = (tkey, state_representation, name)
            score = self._score_cache.get(cache_key)
            if score is None:
                scores = self.memory.tables.get(tkey, {}).get(state_representation)
                if not scores:
                    scores = self.memory.tables.get(None, {}).get(state_representation)
                score = 0.0 if not scores else float(scores.get(name, 0.0))
                if len(self._score_cache) >= self._cache_max:
                    self._score_cache.clear()
                self._score_cache[cache_key] = score

            if best_score is None or score > best_score:
                best = (name, expert, tkey, act)
                best_score = score

        if best is None:
            return self.fallback_agent.choose_action(state_view)

        return best[3]

    def _dice_counts_feature(self, dice):
        c = Counter(int(d) for d in dice)
        return tuple(int(c.get(face, 0)) for face in range(1, 7))

    def _get_state_representation(self, state_view):
        public_state = state_view['public']
        my_dice = state_view['my_dice']
        my_player_id = state_view.get('player_id', 0)

        last_bid = public_state.last_bid
        bid_repr = (last_bid.quantity, last_bid.face) if last_bid else (0, 0)

        my_dice_feat = self._dice_counts_feature(my_dice)

        my_dice_count = int(public_state.dice_counts[my_player_id])
        opponent_dice_counts = [int(count) for i, count in enumerate(public_state.dice_counts) if i != my_player_id]
        player_dice_counts_repr = (my_dice_count, tuple(sorted(opponent_dice_counts)))

        total_dice = int(sum(public_state.dice_counts))
        bid_qty = int(bid_repr[0])
        bid_qty_bucket = 0 if total_dice <= 0 else min(20, int(round((bid_qty / total_dice) * 20)))

        turn_index = int(getattr(public_state, "turn_index", 0) or 0)
        turn_bucket = min(15, turn_index)

        bid_depth = len(getattr(public_state, "bid_history", []) or [])
        bid_depth_bucket = min(15, int(bid_depth))

        is_opening = 1 if last_bid is None else 0

        config = state_view.get("config")
        ones_wild = 1 if bool(getattr(config, "ones_wild", False)) else 0

        return (
            my_dice_feat,
            player_dice_counts_repr,
            bid_repr,
            bid_qty_bucket,
            turn_bucket,
            bid_depth_bucket,
            is_opening,
            ones_wild,
        )
