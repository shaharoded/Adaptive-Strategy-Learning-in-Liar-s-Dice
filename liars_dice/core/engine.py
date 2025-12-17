import random
from typing import Dict
from dataclasses import replace

from .config import GameConfig
from .state import PlayerState, PublicState, GameState
from .dice import roll_n
from .actions import BidAction, CallLiarAction, Action
from .bid import Bid
from .rules import count_matches


class IllegalMoveError(Exception):
    pass

class GameEngine:
    def __init__(self, config: GameConfig):
        self.config = config
        rng_seed = config.rng_seed
        self.rng = random.Random(rng_seed)
        p0 = PlayerState(player_id=0, num_dice=config.dice_distribution[0])
        p1 = PlayerState(player_id=1, num_dice=config.dice_distribution[1])
        public = PublicState()
        self.state = GameState(config=config, players=(p0, p1), public=public)
        self._events = []

    # Events are simple dicts for now
    def _emit(self, event: Dict):
        self._events.append(event)

    def start_new_round(self) -> None:
        # roll dice for each player
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

    def get_view(self, player_id: int):
        # Return a simple view: public + private dice for player
        p = self.state.players[player_id]
        return {
            "player_id": player_id,
            "public": self.state.public,
            "my_dice": tuple(p.private_dice),
        }

    def apply_action(self, player_id: int, action: Action) -> None:
        if self.state.public.status != "BIDDING":
            raise IllegalMoveError("Game is not in bidding state")
        if player_id != self.state.public.current_player:
            raise IllegalMoveError("Not player's turn")

        if isinstance(action, BidAction):
            bid = action.bid
            bid.validate(self.config)
            if not bid.is_higher_than(self.state.public.last_bid, self.config):
                raise IllegalMoveError("Bid is not higher than last bid")
            self.state.public.last_bid = bid
            self.state.public.bid_history.append(bid)
            self.state.public.turn_index += 1
            self.state.public.current_player = 1 - self.state.public.current_player
            self._emit({"type": "BidPlaced", "player": player_id, "bid": (bid.quantity, bid.face)})

        elif isinstance(action, CallLiarAction):
            # resolve immediately
            self._emit({"type": "LiarCalled", "caller": player_id, "last_bid": self.state.public.last_bid})
            self._resolve_call(caller_id=player_id)
        else:
            raise IllegalMoveError("Unknown action")

    def _resolve_call(self, caller_id: int) -> None:
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
        return self.state.public.status == "ENDED"

    def get_events(self):
        return list(self._events)

