import random

from .base import Agent
from ..core.bid import Bid
from ..core.actions import BidAction, CallLiarAction
from . import register_agent


@register_agent("random")
class RandomAgent(Agent):
    """
    A configurable agent that plays Liar's Dice by making random valid bids, calling liar when the opponent's bid seems impossible,
    and probabilistically calling liar more often as the round progresses. Parameters control risk and raise style.
    """
    def __init__(self,
                 rng=None,
                 base_call_prob=0.10,
                 extra_per_turn=0.02,
                 max_call_prob=0.9,
                 raise_amount=1,
                 allow_different_face=True,
                 prob_keep_same_face=0.7,
                 extra_liar_prob_no_face=0.05):
        """
        Args:
            rng: Optional random number generator.
            base_call_prob: Initial probability to call liar (float 0-1).
            extra_per_turn: Probability increase per turn (float).
            max_call_prob: Maximum probability to call liar (float 0-1).
            raise_amount: How much to increase quantity when raising (int >=1).
            allow_different_face: If True, may raise with a different face; if False, always keeps same face.
            prob_keep_same_face: Probability to keep the same face when raising (float 0-1).
            extra_liar_prob_no_face: Extra probability to call liar if agent holds none of the face in question (float 0-1).
        """
        self.rng = rng or random.Random()
        self.base_call_prob = base_call_prob
        self.extra_per_turn = extra_per_turn
        self.max_call_prob = max_call_prob
        self.raise_amount = raise_amount
        self.allow_different_face = allow_different_face
        self.prob_keep_same_face = prob_keep_same_face
        self.extra_liar_prob_no_face = extra_liar_prob_no_face

    def choose_action(self, view):
        """
        Decide the next action based on the current view.
        This agent opens with a random valid bid, calls liar deterministically if the last bid is impossible,
        otherwise calls liar with increasing probability as the round progresses, and otherwise makes a modest raise.
        Args:
            view (dict): Player view with keys 'public', 'my_dice', and optionally 'config'.
        Returns:
            Action: The action to take (BidAction or CallLiarAction).
        """
        public = view.get("public") if hasattr(view, "get") else view["public"]
        my_dice = tuple(view.get("my_dice", ())) if hasattr(view, "get") else tuple(view["my_dice"])
        last = public.last_bid if public is not None else None

        # Determine total dice in game
        config = view.get("config") if hasattr(view, "get") else None
        if config is not None and getattr(config, "dice_distribution", None):
            estimated_total = sum(config.dice_distribution)
        else:
            estimated_total = getattr(config, "total_dice", 5) * getattr(config, "num_players", 1)
        max_qty = estimated_total

        # If no last bid, open with a random but valid bid
        if last is None:
            q = self.rng.randint(1, max_qty)
            f = self.rng.randint(1, 6)
            return BidAction(Bid(q, f))

        # Guard-rail 1: call liar deterministically if the bid is impossible
        if self.call_liar_deterministic(my_dice, last, estimated_total):
            return CallLiarAction()

        # Guard-rail 2: increase chance of calling liar as the bidding goes on
        turn_index = getattr(public, "turn_index", 0) or 0
        extra = min(0.5, turn_index * self.extra_per_turn)
        call_prob = self.base_call_prob + extra

        # Slightly weight toward calling if we hold none of the face in question
        if self.my_count_of_face(my_dice, last.face) == 0:
            call_prob += self.extra_liar_prob_no_face

        # Clamp final probability
        call_prob = min(self.max_call_prob, call_prob)

        # Probabilistic call
        if self.rng.random() < call_prob:
            return CallLiarAction()

        # Otherwise make a raise
        q = min(last.quantity + self.raise_amount, max_qty)
        if self.allow_different_face:
            # Prefer keeping same face, but allow switching sometimes
            f = last.face if self.rng.random() < self.prob_keep_same_face else self.rng.randint(1, 6)
        else:
            f = last.face
        return BidAction(Bid(q, f))


# Example subclasses for different personalities
@register_agent("random_cautious")
class CautiousRandomAgent(RandomAgent):
    """A cautious agent: calls liar less often, raises by 1, never changes face."""
    def __init__(self, rng=None):
        super().__init__(rng=rng, base_call_prob=0.05, extra_per_turn=0.01, max_call_prob=0.5, raise_amount=1, allow_different_face=False)

@register_agent("random_aggressive")
class AggressiveRandomAgent(RandomAgent):
    """An aggressive agent: calls liar more often, raises by 2, may change face."""
    def __init__(self, rng=None):
        super().__init__(rng=rng, base_call_prob=0.20, extra_per_turn=0.05, max_call_prob=0.95, raise_amount=2, allow_different_face=True)

@register_agent("random_facefixed")
class FaceFixedRandomAgent(RandomAgent):
    """Always raises with the same face, never changes face."""
    def __init__(self, rng=None):
        super().__init__(rng=rng, allow_different_face=False)

@register_agent("random_facerandom")
class FaceRandomRandomAgent(RandomAgent):
    """Always allows raising with a different face (randomly)."""
    def __init__(self, rng=None):
        super().__init__(rng=rng, allow_different_face=True)
