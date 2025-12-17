import random
from .base import Agent
from ..core.bid import Bid
from ..core.actions import BidAction, CallLiarAction


class RandomAgent(Agent):
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose_action(self, view):
        public = view["public"]
        last = public.last_bid
        max_qty = 5  # fallback
        # If view includes config, prefer that
        if hasattr(view, "get") and view.get("public") is not None:
            # public may be a PublicState; try to read total dice from surrounding context if present
            # Many views won't include config; we keep a safe default
            pass

        if last is None:
            # make a random opening bid within bounds
            q = self.rng.randint(1, max_qty)
            f = self.rng.randint(1, 6)
            return BidAction(Bid(q, f))

        # small chance to call liar
        if self.rng.random() < 0.1:
            return CallLiarAction()

        # otherwise raise quantity by 1 (clamp to max_qty)
        q = min(last.quantity + 1, max_qty)
        # pick a face (try to keep same face or random)
        f = last.face if self.rng.random() < 0.7 else self.rng.randint(1, 6)
        return BidAction(Bid(q, f))
