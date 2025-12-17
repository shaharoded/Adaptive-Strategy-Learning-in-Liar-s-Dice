import random
from .base import Agent
from ..core.bid import Bid
from ..core.actions import BidAction, CallLiarAction


class RandomAgent(Agent):
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose_action(self, view):
        # view is expected to be a dict with keys: public, my_dice, (optional) config
        public = view.get("public") if hasattr(view, "get") else view["public"]
        my_dice = tuple(view.get("my_dice", ())) if hasattr(view, "get") else tuple(view["my_dice"])  # player's private dice
        last = None
        if public is not None:
            last = public.last_bid

        # Try to get total_dice from an attached config if present; otherwise fallback to common default
        config = view.get("config") if hasattr(view, "get") else None
        estimated_total = getattr(config, "total_dice", None) or 5
        max_qty = estimated_total

        # helper: count how many of a face we have in our private dice
        def my_count_of_face(face: int) -> int:
            return sum(1 for d in my_dice if d == face)

        # If no last bid, open with a random but valid bid
        if last is None:
            q = self.rng.randint(1, max_qty)
            f = self.rng.randint(1, 6)
            return BidAction(Bid(q, f))

        # Guard-rail 1: if the last bid is impossible given our dice and the maximum possible opponent dice,
        # call liar deterministically. We assume the opponent can have at most (estimated_total - len(my_dice)) dice.
        opponent_max = max(0, estimated_total - len(my_dice))
        if last is not None:
            my_count = my_count_of_face(last.face)
            # if even with all opponent dice showing the face the bid cannot be met, it's certainly a lie
            if my_count + opponent_max < last.quantity:
                return CallLiarAction()

        # Guard-rail 2: increase chance of calling liar as the bidding goes on (more turns increases impatience/pressure)
        base_call_prob = 0.10
        # scale increase per turn; clamp extra probability so we don't become too aggressive
        turn_index = getattr(public, "turn_index", 0) or 0
        extra_per_turn = 0.02
        extra = min(0.5, turn_index * extra_per_turn)
        call_prob = base_call_prob + extra

        # Slightly weight toward calling if we hold none of the face in question
        if last is not None:
            if my_count_of_face(last.face) == 0:
                call_prob += 0.05

        # Clamp final probability
        call_prob = min(0.9, call_prob)

        # Probabilistic call
        if self.rng.random() < call_prob:
            return CallLiarAction()

        # Otherwise make a modest raise: increase quantity by 1 but cap at estimated total dice
        q = min(last.quantity + 1, max_qty)
        # Prefer keeping same face, but allow switching sometimes
        f = last.face if self.rng.random() < 0.7 else self.rng.randint(1, 6)
        return BidAction(Bid(q, f))
