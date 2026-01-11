import pickle
from collections import defaultdict
import os
import tempfile
from typing import Any, Dict, Optional


class Memory:
    SCHEMA_VERSION = 3

    def __init__(self, memory_file='adapter_agent_memory.pkl'):
        self.memory_file = memory_file
        # tables[table_key] -> state_key -> agent_name -> score
        # table_key=None means "all actions" (legacy).
        self.tables = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # optional metadata for debugging/compat
        self.meta: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "tables": [None, "Bid", "CallLiar"],
        }

    def update(self, state, agent_name: str, weight: float = 1.0, table_key: Optional[str] = None):
        self.tables[table_key][state][agent_name] += float(weight)

    def update_scores(self, state, winning_agent, weight: float = 1.0):
        # Backwards compatible alias (updates global table)
        self.update(state, winning_agent, weight=weight, table_key=None)

    def get_best_agent(self, state, table_key: Optional[str] = None, smoothing: float = 1.0):
        """
        Return best agent for the given state.

        smoothing: small prior added to each known agent score within this state to avoid extreme decisions
                  on very few samples.
        """
        agent_scores = self.tables[table_key].get(state)
        if not agent_scores:
            # fallback to global table if specific table missing
            if table_key is not None:
                agent_scores = self.tables[None].get(state)
        if not agent_scores:
            return None

        if smoothing and smoothing > 0:
            # apply additive smoothing across the agents seen for this state
            best_agent = None
            best_score = None
            for agent, score in agent_scores.items():
                s = float(score) + float(smoothing)
                if best_score is None or s > best_score:
                    best_agent, best_score = agent, s
            return best_agent

        return max(agent_scores, key=agent_scores.get)

    def save(self, path=None):
        """Saves the memory to a file (atomic write)."""
        filepath = path or self.memory_file
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "meta": dict(self.meta),
            "tables": {
                str(k): {state: dict(agent_scores) for state, agent_scores in v.items()}
                for k, v in self.tables.items()
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_adapter_memory_", suffix=".pkl",
            dir=os.path.dirname(os.path.abspath(filepath)) or "."
        )
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, filepath)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def load(self):
        if not os.path.exists(self.memory_file):
            return

        with open(self.memory_file, 'rb') as f:
            obj = pickle.load(f)

        # Formats:
        # v3+: {tables: {"None": {state:{agent:score}}, "Bid": ...}}
        # v2:  {decision_matrix: {state:{agent:score}}}
        # v1:  {state:{agent:score}}

        self.tables = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        if isinstance(obj, dict) and "tables" in obj:
            self.meta = obj.get("meta", {})
            raw_tables = obj.get("tables", {}) or {}
            for raw_key, states in raw_tables.items():
                table_key = None if raw_key == "None" else raw_key
                for state, agent_scores in (states or {}).items():
                    for agent, score in (agent_scores or {}).items():
                        self.tables[table_key][state][agent] = float(score)
            return

        if isinstance(obj, dict) and "decision_matrix" in obj:
            self.meta = obj.get("meta", {})
            raw = obj.get("decision_matrix", {})
        else:
            raw = obj

        for state, agent_scores in (raw or {}).items():
            for agent, score in (agent_scores or {}).items():
                self.tables[None][state][agent] = float(score)

    def stats(self):
        """Small helper for debugging."""
        per_table = {}
        for k, states in self.tables.items():
            num_states = len(states)
            num_entries = sum(len(v) for v in states.values())
            per_table[str(k)] = {"states": num_states, "entries": num_entries}
        return per_table
