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
    
    def is_bid_possible(self, bid, my_dice, total_dice, ones_wild=False, faces=None):
        """
        Returns True if the bid is possible given my_dice and total_dice.
        """
        if bid.quantity > total_dice:
            return False
        # Count how many dice of the bid face (or ones if wild)
        count = my_dice.count(bid.face)
        if ones_wild and faces is not None and bid.face != 1:
            count += my_dice.count(1)
        # If I have 0 of that face (and ones if wild), it's still possible, but less likely
        # Only impossible if quantity > total_dice
        return bid.quantity <= total_dice

    def is_last_bid_impossible(self, last_bid, my_dice, total_dice, ones_wild=False, faces=None):
        if last_bid is None:
            return False
        # If the last bid requires more dice than possible, it's impossible
        if last_bid.quantity > total_dice:
            return True
        # If I have 0 of the face (and ones if wild), and the bid quantity is more than total_dice - my_dice_count, it's impossible
        count = my_dice.count(last_bid.face)
        if ones_wild and faces is not None and last_bid.face != 1:
            count += my_dice.count(1)
        # If I have none, but the bid is for all dice, it's impossible
        return last_bid.quantity > total_dice


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
        total_dice = self.get_num_dice(view)
        ones_wild = getattr(config, 'ones_wild', False)
        faces = config.faces
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
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
        total_dice = self.get_num_dice(view)
        ones_wild = getattr(config, 'ones_wild', False)
        faces = config.faces
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        # Try all valid higher bids (by quantity or face), prefer higher quantities first, only possible bids
        for q in range(total_dice, last_bid.quantity, -1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
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
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
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
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
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
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        # Generate all valid higher bids
        candidates = []
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
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
        total_dice = self.get_num_dice(view)
        ones_wild = getattr(config, 'ones_wild', False)
        faces = config.faces
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        # Mirror: only raise quantity for the same face as last bid
        for q in range(last_bid.quantity + 1, total_dice + 1):
            candidate = Bid(q, last_bid.face)
            if self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
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
        total_dice = self.get_num_dice(view)
        ones_wild = getattr(config, 'ones_wild', False)
        faces = config.faces
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        # Otherwise, bid up minimally
        try:
            next_bid = Bid(last_bid.quantity + 1, last_bid.face)
            if self.is_bid_possible(next_bid, my_dice, total_dice, ones_wild, faces):
                next_bid.validate(config)
                return BidAction(next_bid)
        except Exception:
            pass
        return CallLiarAction()


# RandomFaceAgent: always bids the next legal bid with a random face
@register_agent("randomface")
class RandomFaceAgent(HeuristicAgent):
    """
    RandomFaceAgent:
    - Always bids the next legal bid with a random face, regardless of the last bid’s face.
    - Calls liar if no valid bid is possible.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        candidates = []
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                    try:
                        candidate.validate(config)
                        candidates.append(candidate)
                    except Exception:
                        continue
        if not candidates:
            return CallLiarAction()
        return BidAction(random.choice(candidates))


# SafeFaceAgent: prefers to bid faces it has in hand
@register_agent("safeface")
class SafeFaceAgent(HeuristicAgent):
    """
    SafeFaceAgent:
    - Prefers to bid faces it has in hand, picking the minimal valid raise for those faces.
    - If no such bid is possible, falls back to any minimal valid raise.
    - Calls liar if no valid bid is possible.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        if last_bid is None:
            # Pick the face in hand with highest count
            from collections import Counter
            counts = Counter(my_dice)
            face, _ = counts.most_common(1)[0]
            return BidAction(Bid(1, face))
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        # Prefer faces in hand
        for q in range(last_bid.quantity, total_dice + 1):
            for f in set(my_dice):
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        # Fallback: any minimal valid raise
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()


# OnesAreWildAgent: prefers to bid on ones if ones are wild
@register_agent("onesarewild")
class OnesAreWildAgent(HeuristicAgent):
    """
    OnesAreWildAgent:
    - If ones are wild, prefers to bid on ones, otherwise acts like SafeFaceAgent.
    - Calls liar if no valid bid is possible.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        ones_wild = getattr(config, 'ones_wild', False)
        if last_bid is None:
            if ones_wild:
                return BidAction(Bid(1, 1))
            else:
                from collections import Counter
                counts = Counter(my_dice)
                face, _ = counts.most_common(1)[0]
                return BidAction(Bid(1, face))
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        # Prefer ones if wild
        if ones_wild:
            for q in range(last_bid.quantity, total_dice + 1):
                candidate = Bid(q, 1)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        # Otherwise, fallback to SafeFaceAgent logic
        for q in range(last_bid.quantity, total_dice + 1):
            for f in set(my_dice):
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()


# BluffingAgent: sometimes bids on faces not in hand
@register_agent("bluffing")
class BluffingAgent(HeuristicAgent):
    """
    BluffingAgent:
    - With probability bluff_chance, bids on a face not in hand (if possible), otherwise acts like SafeFaceAgent.
    - Calls liar if no valid bid is possible.
    """
    def __init__(self, bluff_chance=0.2):
        super().__init__()
        self.bluff_chance = bluff_chance

    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        not_in_hand = [f for f in faces if f not in my_dice]
        if last_bid is None:
            if not_in_hand and random.random() < self.bluff_chance:
                return BidAction(Bid(1, random.choice(not_in_hand)))
            else:
                from collections import Counter
                counts = Counter(my_dice)
                face, _ = counts.most_common(1)[0]
                return BidAction(Bid(1, face))
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        # Try bluff
        if not_in_hand and random.random() < self.bluff_chance:
            for q in range(last_bid.quantity, total_dice + 1):
                for f in not_in_hand:
                    candidate = Bid(q, f)
                    if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                        try:
                            candidate.validate(config)
                            return BidAction(candidate)
                        except Exception:
                            continue
        # Otherwise, SafeFaceAgent logic
        for q in range(last_bid.quantity, total_dice + 1):
            for f in set(my_dice):
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()


# ThresholdLiarAgent: calls liar if bid exceeds a threshold
@register_agent("thresholdliar")
class ThresholdLiarAgent(HeuristicAgent):
    """
    ThresholdLiarAgent:
    - Calls liar if the last bid's quantity exceeds a threshold (default: half the total dice, rounded up).
    - Otherwise, makes a minimal valid raise (by quantity or face).
    """
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        threshold = self.threshold or ((total_dice + 1) // 2)
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible, call liar
        if self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        if last_bid.quantity > threshold:
            return CallLiarAction()
        # Otherwise, minimal valid raise
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid) and self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()


# ChaoticAgent: randomly chooses between minimal raise, maximal raise, or calling liar
@register_agent("chaotic")
class ChaoticAgent(HeuristicAgent):
    """
    ChaoticAgent:
    - On each turn, randomly chooses to make a minimal raise, maximal raise, or call liar, regardless of state.
    """
    def __init__(self, allow_impossible=False):
        super().__init__()
        self.allow_impossible = allow_impossible

    def choose_action(self, view):
        from random import choice
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        ones_wild = getattr(config, 'ones_wild', False)
        # If last bid is impossible and not allowed, call liar
        if not self.allow_impossible and self.is_last_bid_impossible(last_bid, my_dice, total_dice, ones_wild, faces):
            return CallLiarAction()
        candidates = []
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    if self.allow_impossible or self.is_bid_possible(candidate, my_dice, total_dice, ones_wild, faces):
                        try:
                            candidate.validate(config)
                            candidates.append(candidate)
                        except Exception:
                            continue
        options = []
        if candidates:
            min_bid = min(candidates, key=lambda b: (b.quantity, b.face))
            max_bid = max(candidates, key=lambda b: (b.quantity, b.face))
            options.extend([BidAction(min_bid), BidAction(max_bid)])
        options.append(CallLiarAction())
        return choice(options)

# Register both variants
@register_agent("chaotic_safe")
class ChaoticSafeAgent(ChaoticAgent):
    def __init__(self):
        super().__init__(allow_impossible=False)

@register_agent("chaotic_unsafe")
class ChaoticUnsafeAgent(ChaoticAgent):
    def __init__(self):
        super().__init__(allow_impossible=True)

# AlternatorAgent: alternates between calling liar and minimal raise
@register_agent("alternator")
class AlternatorAgent(HeuristicAgent):
    """
    AlternatorAgent:
    - Alternates between calling liar and making a minimal valid raise, regardless of state.
    """
    def __init__(self):
        super().__init__()
        self.last_action_was_liar = False

    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        if last_bid is None:
            self.last_action_was_liar = False
            return BidAction(Bid(1, random.choice(my_dice)))
        if self.last_action_was_liar:
            # Make minimal raise
            for q in range(last_bid.quantity, total_dice + 1):
                for f in faces:
                    candidate = Bid(q, f)
                    if candidate.is_higher_than(last_bid):
                        try:
                            candidate.validate(config)
                            self.last_action_was_liar = False
                            return BidAction(candidate)
                        except Exception:
                            continue
            self.last_action_was_liar = False
            return CallLiarAction()
        else:
            self.last_action_was_liar = True
            return CallLiarAction()

# CycleFaceAgent: always bids the next face in sequence, raising quantity as needed
@register_agent("cycleface")
class CycleFaceAgent(HeuristicAgent):
    """
    CycleFaceAgent:
    - Always bids the next face in sequence (wrapping around), raising quantity as needed.
    - Calls liar if no valid bid is possible.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = list(config.faces)
        total_dice = self.get_num_dice(view)
        if last_bid is None:
            return BidAction(Bid(1, faces[0]))
        # Find next face in sequence
        try:
            idx = faces.index(last_bid.face)
        except ValueError:
            idx = 0
        for offset in range(1, len(faces) + 1):
            next_face = faces[(idx + offset) % len(faces)]
            for q in range(last_bid.quantity, total_dice + 1):
                candidate = Bid(q, next_face)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()

# ParityAgent: calls liar if last bid's quantity is even, else raises minimally
@register_agent("parity")
class ParityAgent(HeuristicAgent):
    """
    ParityAgent:
    - If the last bid's quantity is even, calls liar; if odd, raises minimally.
    """
    def choose_action(self, view):
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        if last_bid.quantity % 2 == 0:
            return CallLiarAction()
        # Otherwise, minimal valid raise
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()

# RandomThresholdAgent: picks a random threshold at the start of each game
@register_agent("randomthreshold")
class RandomThresholdAgent(HeuristicAgent):
    """
    RandomThresholdAgent:
    - Picks a random threshold at the start of each game and calls liar if the bid exceeds it.
    - Otherwise, makes a minimal valid raise.
    """
    def __init__(self):
        super().__init__()
        self.threshold = None

    def choose_action(self, view):
        import math
        my_dice = self.get_my_dice(view)
        last_bid = self.get_last_bid(view)
        config = self.get_config(view)
        faces = config.faces
        total_dice = self.get_num_dice(view)
        if self.threshold is None or last_bid is None:
            # Pick a new threshold at the start of each game
            self.threshold = random.randint(math.ceil(total_dice / 3), total_dice)
        if last_bid is None:
            return BidAction(Bid(1, random.choice(my_dice)))
        if last_bid.quantity > self.threshold:
            return CallLiarAction()
        # Otherwise, minimal valid raise
        for q in range(last_bid.quantity, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                        return BidAction(candidate)
                    except Exception:
                        continue
        return CallLiarAction()