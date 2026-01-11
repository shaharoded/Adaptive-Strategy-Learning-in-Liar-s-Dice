from .base import Agent, UntrainedAgentException
from adapter_agent.memory import Memory
from liars_dice.agents.random_agent import RandomAgent
from liars_dice.agents.bayesian_agent import BayesianAgent
from liars_dice.agents.heuristic_agent import ConservativeAgent
from liars_dice.agents.nash_agent import NashCFRAgent

from collections import Counter


class AdapterAgent(Agent):
    def __init__(self, memory_file='adapter_agent_memory.pkl'):
        super().__init__()
        self.memory = Memory(memory_file)
        self.memory.load()
        self.agents = self._load_agents()
        self.fallback_agent = RandomAgent()

        self._cache = {}
        self._cache_max = 50_000

    def _load_agents(self):
        agents = {
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
        state_representation = self._get_state_representation(state_view)

        # Determine which table to consult first based on what actions are even legal.
        # If there's no last bid yet, CallLiar is illegal, so use Bid table.
        public = state_view["public"]
        table_key = "Bid" if public.last_bid is None else None

        cache_key = (table_key, state_representation)
        cached = self._cache.get(cache_key)
        if cached is None:
            # Prefer action-specific if we know which regime we're in, else global.
            # During bidding (last_bid exists) both actions are possible, so we consult a global table.
            best = self.memory.get_best_agent(state_representation, table_key=table_key)
            if best is None:
                best = self.memory.get_best_agent(state_representation, table_key=None)
            cached = best
            if len(self._cache) >= self._cache_max:
                self._cache.clear()
            self._cache[cache_key] = cached

        best_agent_name = cached

        if best_agent_name and best_agent_name in self.agents:
            agent_to_use = self.agents[best_agent_name]
        else:
            agent_to_use = self.fallback_agent

        return agent_to_use.choose_action(state_view)

    def _dice_counts_feature(self, dice):
        c = Counter(int(d) for d in dice)
        return tuple(int(c.get(face, 0)) for face in range(1, 7))

    def _get_state_representation(self, state_view):
        public_state = state_view['public']
        my_dice = state_view['my_dice']
        my_player_id = state_view['player_id']

        last_bid = public_state.last_bid
        bid_repr = (last_bid.quantity, last_bid.face) if last_bid else None

        my_dice_feat = self._dice_counts_feature(my_dice)

        my_dice_count = public_state.dice_counts[my_player_id]
        opponent_dice_counts = [count for i, count in enumerate(public_state.dice_counts) if i != my_player_id]
        player_dice_counts_repr = (my_dice_count, tuple(sorted(opponent_dice_counts)))

        total_dice = sum(public_state.dice_counts)
        if bid_repr is None or total_dice <= 0:
            pressure = 0
        else:
            pressure = min(10, int(round((bid_repr[0] / total_dice) * 10)))

        turn_index = int(getattr(public_state, "turn_index", 0) or 0)
        turn_bucket = min(10, turn_index)

        return (
            my_dice_feat,
            player_dice_counts_repr,
            bid_repr,
            pressure,
            turn_bucket,
        )
