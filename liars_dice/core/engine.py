
"""
engine.py
Implements the GameEngine class, which manages game state, applies actions, enforces rules, and emits events.
Related modules:
- config.py: GameConfig is used to configure the engine.
- state.py: GameState, PlayerState, PublicState hold all game data.
- actions.py: Actions are applied to update state.
- bid.py: Bid validation and ordering.
- rules.py: Helpers for counting dice matches and wild ones logic.
"""

import random
from typing import Dict

from .config import GameConfig
from .state import PlayerState, PublicState, GameState
from .dice import roll_n
from .actions import BidAction, CallLiarAction, Action
from .rules import count_matches


class IllegalMoveError(Exception):
    """
    Raised when an illegal action is attempted (invalid move, wrong turn, etc).
    """
    pass


class GameEngine:
    """
    Main state machine for Liar's Dice. Manages game state, applies actions, enforces legality, and emits events.
    Interacts with agents via get_view and apply_action.
    """
    def __init__(self, config: GameConfig):
        """
        Initialize a new game engine with the given configuration.
        Args:
            config (GameConfig): Game configuration.
        """
        self.config = config
        rng_seed = config.rng_seed
        self.rng = random.Random(rng_seed)
        # Determine per-player dice distribution: prefer explicit dice_distribution, otherwise use total_dice per player
        if getattr(config, "dice_distribution", None):
            dist = tuple(config.dice_distribution)
        else:
            # interpret total_dice as per-player count
            dist = tuple(config.total_dice for _ in range(config.num_players))
        # Ensure we have enough entries for the configured number of players
        if len(dist) < config.num_players:
            dist = tuple(dist[i % len(dist)] for i in range(config.num_players))

        # For now the state expects exactly two players; create player states using distribution
        p0 = PlayerState(player_id=0, num_dice=dist[0])
        p1 = PlayerState(player_id=1, num_dice=dist[1])
        public = PublicState()
        self.state = GameState(config=config, players=(p0, p1), public=public)
        self._events = []
        # turn_log will contain per-turn snapshots that can be serialized to JSON
        self.turn_log = []

    # Events are simple dicts for now
    def _emit(self, event: Dict):
        """
        Internal: Record an event (dict) for later retrieval.
        """
        self._events.append(event)

    def pop_events(self):
        """
        Return and clear all emitted events since last call.
        Returns:
            list[dict]: List of event dicts.
        """
        ev = list(self._events)
        self._events.clear()
        return ev

    def _snapshot(self, actor: int = None, action: Dict = None):
        """
        Internal: Create a snapshot of the current state for logging/replay.
        Args:
            actor (int|None): Player who made the action.
            action (dict|None): Action that produced this state.
        Returns:
            dict: Snapshot of state.
        """
        # shallow-copy players' public-facing fields and private dice
        players_snapshot = []
        for p in self.state.players:
            players_snapshot.append({
                "player_id": p.player_id,
                "num_dice": p.num_dice,
                "private_dice": list(p.private_dice),
                "agent_id": p.agent_id,
            })

        last_bid = self.state.public.last_bid
        last_bid_ser = None if last_bid is None else (last_bid.quantity, last_bid.face)
        bid_history_ser = [ (b.quantity, b.face) for b in self.state.public.bid_history ]

        public_snapshot = {
            "round_index": self.state.public.round_index,
            "turn_index": self.state.public.turn_index,
            "current_player": self.state.public.current_player,
            "last_bid": last_bid_ser,
            "bid_history": bid_history_ser,
            "status": self.state.public.status,
            "winner": self.state.public.winner,
            "loser": self.state.public.loser,
        }

        snap = {
            "actor": actor,
            "action": action,
            "public": public_snapshot,
            "players": players_snapshot,
        }
        self.turn_log.append(snap)
        return snap

    def start_new_round(self) -> None:
        """
        Start a new round: roll dice, reset public state, emit initial events.
        """
        p0, p1 = self.state.players
        p0.private_dice = roll_n(p0.num_dice, self.rng)
        p1.private_dice = roll_n(p1.num_dice, self.rng)
        self.state.public.status = "BIDDING"
        self.state.public.round_index += 1
        self.state.public.turn_index = 0
        self.state.public.current_player = 0
        self.state.public.last_bid = None
        self.state.public.bid_history = []
        self.state.public.winner = None
        self.state.public.loser = None
        self._emit({"type": "RoundStarted", "round": self.state.public.round_index})
        self._emit({"type": "DiceRolled", "player0": p0.private_dice.copy(), "player1": p1.private_dice.copy()})
        # snapshot initial state of the round (no actor/action)
        self._snapshot(actor=None, action=None)

    def get_view(self, player_id: int):
        """
        Get a player-specific view of the game state (public info + private dice).
        Args:
            player_id (int): Player index (0 or 1).
        Returns:
            dict: Player view for agent decision-making.
        """
        p = self.state.players[player_id]
        return {
            "player_id": player_id,
            "public": self.state.public,
            "my_dice": tuple(p.private_dice),
            "config": self.config,
        }

    def apply_action(self, player_id: int, action: Action) -> None:
        """
        Apply an action for the given player, updating state and emitting events.
        Args:
            player_id (int): Player index (0 or 1).
            action (Action): The action to apply (BidAction or CallLiarAction).
        Raises:
            IllegalMoveError: If action is invalid or not player's turn.
        """
        if self.state.public.status != "BIDDING":
            raise IllegalMoveError("Game is not in bidding state")
        if player_id != self.state.public.current_player:
            raise IllegalMoveError("Not player's turn")

        # We'll serialize the action for logging after it's applied
        action_ser = None

        if isinstance(action, BidAction):
            bid = action.bid
            bid.validate(self.config)
            if not bid.is_higher_than(self.state.public.last_bid):
                raise IllegalMoveError("Bid is not higher than last bid")
            self.state.public.last_bid = bid
            self.state.public.bid_history.append(bid)
            self.state.public.turn_index += 1
            self.state.public.current_player = 1 - self.state.public.current_player
            self._emit({"type": "BidPlaced", "player": player_id, "bid": (bid.quantity, bid.face)})
            action_ser = {"type": "Bid", "bid": (bid.quantity, bid.face)}

        elif isinstance(action, CallLiarAction):
            # resolve immediately
            self._emit({"type": "LiarCalled", "caller": player_id, "last_bid": self.state.public.last_bid})
            action_ser = {"type": "CallLiar"}
            self._resolve_call(caller_id=player_id)
        else:
            raise IllegalMoveError("Unknown action")

        # snapshot resulting state and the action that produced it
        self._snapshot(actor=player_id, action=action_ser)

    def _resolve_call(self, caller_id: int) -> None:
        """
        Internal: Resolve a CallLiarAction, reveal dice, determine winner/loser, emit events.
        Args:
            caller_id (int): Player who called liar.
        Raises:
            IllegalMoveError: If no bid to call.
        """
        # reveal dice and determine winner/loser
        all_dice = {0: self.state.players[0].private_dice, 1: self.state.players[1].private_dice}
        last_bid = self.state.public.last_bid
        if last_bid is None:
            # calling a non-existing bid -> treat as illegal
            raise IllegalMoveError("No bid to call")
        match_count = count_matches(all_dice, last_bid.face, self.config.ones_wild)
        was_true = match_count >= last_bid.quantity
        if was_true:
            # caller loses
            loser = caller_id
            winner = 1 - caller_id
        else:
            loser = 1 - caller_id
            winner = caller_id
        self.state.public.winner = winner
        self.state.public.loser = loser
        self.state.public.status = "ENDED"
        self._emit({"type": "DiceRevealed", "all_dice": all_dice})
        self._emit({"type": "RoundEnded", "winner": winner, "loser": loser, "match_count": match_count, "was_true": was_true})

    def is_terminal(self) -> bool:
        """
        Returns True if the game round has ended.
        """
        return self.state.public.status == "ENDED"

    def get_events(self):
        """
        Return all events emitted so far (does not clear).
        Returns:
            list[dict]: List of event dicts.
        """
        return list(self._events)
