from liars_dice.agents.base import Agent, register_agent
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid
import random
from collections import Counter


class HeuristicAgent(Agent):
    """
    Base class for heuristic agents: agents that implements some kind of deterministic strategy. 
    Provides utility methods for subclasses which in turn will use these subclasses to decide on action.
    Utility methods:
        - get_my_dice(view): Returns the agent's dice as a list.
        - get_last_bid(view): Returns the last bid placed (Bid or None).
        - get_config(view): Returns the game config object.
        - get_num_dice(view): Returns the total number of dice in play.
    """
    def __init__(self):
        super().__init__()

    def get_my_dice(self, view):
        return view["my_dice"]

    def get_last_bid(self, view):
        return view["public"].last_bid

    def get_config(self, view):
        return view.get("config")

    def get_num_dice(self, view):
        return sum(view["public"].dice_counts)


@register_agent("conservative")
class ConservativeAgent(HeuristicAgent):
    """
    ConservativeAgent:
    - If no bid has been made, starts with a low bid (quantity 1, face of one of its dice).
    - If the last bid is higher than the number of dice the agent has, calls liar immediately (does not believe the bid is possible).
    - Otherwise, increases the quantity by 1 (keeping the same face) if possible; if not, calls liar.
    This agent is risk-averse and quick to challenge high bids.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        # If no bid, start with a low bid
        if last_bid is None:
            return BidAction(Bid(1, my_dice[0]))
        # If last bid is high, call liar
        if last_bid.quantity > len(my_dice):
            return CallLiarAction()
        # Otherwise, suggest the smallest possible valid raise (by quantity or face)
        faces = config.faces
        for q in range(last_bid.quantity, len(my_dice) + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()


@register_agent("aggressive")
class AggressiveAgent(HeuristicAgent):
    """
    AggressiveAgent:
    - If no bid has been made, starts with a high bid (quantity = number of dice, face is random from own dice).
    - Always tries to increase the quantity by 1, cycling through all possible faces.
    - Only calls liar if no valid higher bid is possible.
    This agent is bold and prefers to keep bidding rather than challenge.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        # If no bid, start with a high bid
        if last_bid is None:
            return BidAction(Bid(len(my_dice), random.choice(my_dice)))
        # Try all valid higher bids (by quantity or face), prefer higher quantities first
        faces = config.faces
        max_qty = self.get_num_dice(view)
        for q in range(max_qty, last_bid.quantity, -1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        # If can't bid higher, reluctantly call liar
        return CallLiarAction()


# Probabilistic agent with configurable raise preference
class ProbabilityAgent(HeuristicAgent):
    """
    ProbabilityAgent:
    - If no bid has been made, starts with a likely bid (quantity 1, face is random from own dice).
    - Estimates the expected count of the last bid's face in all dice.
    - If the last bid's quantity is much higher than expected (expected + 1), calls liar.
    - Otherwise, tries all valid higher bids (by quantity or face).
    - If prefer_maximal is True, picks the maximal valid raise; else, picks the minimal valid raise.
    """
    def __init__(self, prefer_maximal=False):
        super().__init__()
        self.prefer_maximal = prefer_maximal

    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        total_dice = self.get_num_dice(view)
        faces = config.faces
        # If no bid, start with a likely bid
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        # Estimate probability last bid is true
        expected = total_dice / len(faces)
        # If bid is much higher than expected, call liar
        if last_bid.quantity > expected + 1:
            return CallLiarAction()
        # Otherwise, try all valid higher bids (by quantity or face)
        candidates = []
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        candidates.append(candidate)
                    except Exception:
                        continue
        if not candidates:
            return CallLiarAction()
        if self.prefer_maximal:
            chosen = max(candidates, key=lambda b: (b.quantity, b.face))
        else:
            chosen = min(candidates, key=lambda b: (b.quantity, b.face))
        return BidAction(chosen)

# Register both variants
@register_agent("probability_minraise")
class ProbabilityMinRaiseAgent(ProbabilityAgent):
    def __init__(self):
        super().__init__(prefer_maximal=False)

@register_agent("probability_maxraise")
class ProbabilityMaxRaiseAgent(ProbabilityAgent):
    def __init__(self):
        super().__init__(prefer_maximal=True)
    

# Generic raise agent: can prefer minimal or maximal raise
class RaisePreferenceAgent(HeuristicAgent):
    """
    RaisePreferenceAgent:
    - If no bid has been made, starts with a likely bid (quantity 1, face is random from own dice).
    - Otherwise, tries all valid higher bids (by quantity or face).
    - If prefer_maximal is True, picks the maximal valid raise; else, picks the minimal valid raise.
    """
    def __init__(self, prefer_maximal=False):
        super().__init__()
        self.prefer_maximal = prefer_maximal

    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        total_dice = self.get_num_dice(view)
        faces = config.faces
        # If no bid, start with a likely bid
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        # Generate all valid higher bids
        candidates = []
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        candidates.append(candidate)
                    except Exception:
                        continue
        if not candidates:
            return CallLiarAction()
        # Pick minimal or maximal raise
        if self.prefer_maximal:
            chosen = max(candidates, key=lambda b: (b.quantity, b.face))
        else:
            chosen = min(candidates, key=lambda b: (b.quantity, b.face))
        return BidAction(chosen)

# Register both variants
@register_agent("minraise")
class MinRaiseAgent(RaisePreferenceAgent):
    def __init__(self):
        super().__init__(prefer_maximal=False)

@register_agent("maxraise")
class MaxRaiseAgent(RaisePreferenceAgent):
    def __init__(self):
        super().__init__(prefer_maximal=True)



@register_agent("mirror")
class MirrorAgent(HeuristicAgent):
    """
    MirrorAgent:
    - If no bid has been made, starts with a random bid (quantity 1, face is random from own dice).
    - Tries to repeat the last bid (if it's legal and higher than the previous bid).
    - If not possible, increases the quantity by 1 (same face) if possible; if not, calls liar.
    This agent tries to mimic the opponent's last move, otherwise bids up minimally.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        # If no bid, start with a random bid
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        # Mirror: only raise quantity for the same face as last bid
        max_qty = self.get_num_dice(view)
        for q in range(last_bid.quantity + 1, max_qty + 1):
            candidate = Bid(q, last_bid.face)
            try:
                candidate.validate(config)
                return BidAction(candidate)
            except Exception:
                continue
        return CallLiarAction()
        

@register_agent("maxcount")
class MaxCountBidAgent(HeuristicAgent):
    """
    MaxCountBidAgent:
    - On its first bid, always bids the face it has the most of, at the highest count (e.g., if it has three 4s, bids (3, 4)).
    - On subsequent turns, increases the quantity by 1 (same face as last bid) if possible; if not, calls liar.
    This agent always opens with its strongest face and tries to push the count up.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        if last_bid is None:
            # Find the face with the highest count in my dice
            counts = Counter(my_dice)
            face, qty = counts.most_common(1)[0]
            return BidAction(Bid(qty, face))
        # Otherwise, bid up minimally
        try:
            next_bid = Bid(last_bid.quantity + 1, last_bid.face)
            next_bid.validate(config)
            return BidAction(next_bid)
        except Exception:
            return CallLiarAction()
