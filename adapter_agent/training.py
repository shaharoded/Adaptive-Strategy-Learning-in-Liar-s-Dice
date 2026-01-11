import sys
import os
# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from liars_dice.core.engine import GameEngine
from liars_dice.core.config import GameConfig
from liars_dice.agents.random_agent import RandomAgent
from liars_dice.agents.bayesian_agent import BayesianAgent
from liars_dice.agents.heuristic_agent import ConservativeAgent
from liars_dice.agents.nash_agent import NashCFRAgent
from liars_dice.agents.base import UntrainedAgentException
from adapter_agent.memory import Memory


class Trainer:
    def __init__(self, num_games, num_players=2):
        self.num_games = num_games
        self.num_players = num_players
        self.memory = Memory()
        self.agents = self._load_agents()

        # Save memory periodically
        self.checkpoint_interval = 1_000_000
        self.checkpoint_dir = 'adapter_agent_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Reward shaping
        self.gamma = 0.98  # temporal discount for earlier decisions
        self.log_interval = 10_000

        # Learning signal
        self.win_reward = 1.0
        self.lose_penalty = 1.0

    def _load_agents(self):
        # For simplicity, we'll manually instantiate agents.
        # A more robust solution could use dynamic loading.
        agents = {
            "RandomAgent": RandomAgent(),
            "BayesianAgent": BayesianAgent(),
            "HeuristicAgent": ConservativeAgent(),
        }
        # Nash agent might not have trained weights in this repo setup.
        try:
            agents["NashAgent"] = NashCFRAgent()
        except UntrainedAgentException:
            # still include the name, but map to a strong-ish fallback if untrained
            agents["NashAgent"] = BayesianAgent()
        return agents

    def train(self):
        agent_names = list(self.agents.keys())

        for i in range(self.num_games):
            # Rotate agents for each game
            agent_player1_name = agent_names[i % len(agent_names)]
            agent_player2_name = agent_names[(i + 1) % len(agent_names)]

            game_agents = [self.agents[agent_player1_name], self.agents[agent_player2_name]]

            # Simulate a full match using the engine directly (avoids Agent.act mismatch)
            config = GameConfig(num_players=2)
            engine = GameEngine(config)

            # We record EVERY visited decision point as (state_key, actor_id, action_type)
            trajectory = []

            def action_type_of(action_obj) -> str:
                if not isinstance(action_obj, dict):
                    return "Other"
                return action_obj.get("type") or "Other"

            def record_pre_action_state(pid: int):
                # Build a snapshot-like dict for state *before* the action.
                # This guarantees we record all passed states / decision points.
                state_key = self._get_state_representation_from_engine(engine, pid)
                last_snap = engine.turn_log[-1] if engine.turn_log else None
                last_action = None if not last_snap else last_snap.get("action")
                # The action that is about to be taken is unknown; we bucket by what gets applied.
                # We'll fill action_type after apply_action using the post-action snap.
                trajectory.append({
                    "state_key": state_key,
                    "actor": pid,
                    "action_type": None,
                })

            def play_match():
                while True:
                    if any(p.num_dice == 0 for p in engine.state.players):
                        return
                    engine.start_new_round()
                    while not engine.is_terminal():
                        pid = engine.state.public.current_player
                        record_pre_action_state(pid)
                        view = engine.get_view(pid)
                        action = game_agents[pid].choose_action(view)
                        engine.apply_action(pid, action)

                        # After apply_action, engine.turn_log[-1] is the post-action snapshot
                        post = engine.turn_log[-1]
                        atype = action_type_of(post.get("action"))
                        # update the last recorded step's action type
                        trajectory[-1]["action_type"] = atype

                    loser = engine.state.public.loser
                    if loser is not None:
                        engine.state.players[loser].num_dice -= 1

            play_match()

            # winner is whoever still has dice
            winner_id = None
            loser_id = None
            for p in engine.state.players:
                if p.num_dice > 0:
                    winner_id = p.player_id
                else:
                    loser_id = p.player_id

            if winner_id is not None and loser_id is not None:
                winner_agent_name = game_agents[winner_id].__class__.__name__
                loser_agent_name = game_agents[loser_id].__class__.__name__

                T = max(1, len(trajectory))
                for t, step in enumerate(trajectory):
                    actor = step["actor"]
                    state_key = step["state_key"]
                    action_type = step.get("action_type")
                    table_key = action_type if action_type in ("Bid", "CallLiar") else None

                    # Discount earlier decisions
                    w = (self.gamma ** (T - 1 - t))

                    if actor == winner_id:
                        self.memory.update(state_key, winner_agent_name, weight=self.win_reward * w, table_key=table_key)
                        self.memory.update(state_key, winner_agent_name, weight=self.win_reward * w, table_key=None)
                    elif actor == loser_id:
                        self.memory.update(state_key, loser_agent_name, weight=-self.lose_penalty * w, table_key=table_key)
                        self.memory.update(state_key, loser_agent_name, weight=-self.lose_penalty * w, table_key=None)

            if (i + 1) % self.log_interval == 0:
                print(f"Trained {i+1}/{self.num_games} games. Memory: {self.memory.stats()}")

            if (i + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"memory_checkpoint_{(i + 1)}_games.pkl")
                self.memory.save(checkpoint_path)
                print(f"Checkpoint saved at game {i+1} to {checkpoint_path}")

        self.memory.save()
        print("Training complete. Final memory saved.")

    def _dice_counts_feature(self, dice_list):
        """Return a compact 6-tuple of face counts (faces 1..6)."""
        c = Counter(int(d) for d in dice_list)
        return tuple(int(c.get(face, 0)) for face in range(1, 7))

    def _get_state_representation_from_engine(self, engine: GameEngine, player_id: int):
        """State key for a specific player (actor), i.e., information set at decision time."""
        view = engine.get_view(player_id)
        public_state = view["public"]
        my_dice = view["my_dice"]

        last_bid = public_state.last_bid
        bid_repr = (last_bid.quantity, last_bid.face) if last_bid else None

        my_dice_feat = self._dice_counts_feature(my_dice)

        my_dice_count = public_state.dice_counts[player_id]
        opponent_dice_counts = [count for i, count in enumerate(public_state.dice_counts) if i != player_id]
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

    # keep old method in case you still use snapshot-based flows elsewhere
    def _get_state_representation_from_snapshot(self, snap):
        """
        A canonical, generic representation built from engine snapshots.

        - Uses counts-per-face instead of sorted dice (same info, smaller space, more robust).
        - Uses my dice count + sorted opponent dice counts.
        - Includes last bid (quantity, face).
        - Adds a coarse 'pressure' feature: last_bid.quantity relative to total dice.
        """
        public = snap["public"]
        players = snap["players"]

        current_player_id = public["current_player"]
        my_private_dice = players[current_player_id]["private_dice"]
        my_dice_feat = self._dice_counts_feature(my_private_dice)

        my_dice_count = int(players[current_player_id]["num_dice"])
        opponent_dice_counts = [int(p["num_dice"]) for p in players if p["player_id"] != current_player_id]
        player_dice_counts_repr = (my_dice_count, tuple(sorted(opponent_dice_counts)))

        last_bid = public.get("last_bid")
        bid_repr = tuple(last_bid) if last_bid is not None else None

        total_dice = sum(int(p["num_dice"]) for p in players)
        if bid_repr is None or total_dice <= 0:
            pressure = 0
        else:
            pressure = min(10, int(round((bid_repr[0] / total_dice) * 10)))

        turn_index = int(public.get("turn_index", 0) or 0)
        turn_bucket = min(10, turn_index)

        return (
            my_dice_feat,
            player_dice_counts_repr,
            bid_repr,
            pressure,
            turn_bucket,
        )


if __name__ == '__main__':
    # This script can be run to train the adapter agent
    trainer = Trainer(num_games=10_000_000)
    trainer.train()

