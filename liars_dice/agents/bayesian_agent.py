"""liars_dice.agents.bayesian_agent

A Bayesian-flavored Liar's Dice agent.

Strategy:
- Given the current public last bid and our private dice, estimate P(bid is true)
  assuming unknown opponent dice are i.i.d. fair.
- If P(bid true) is below a threshold, call liar.
- Otherwise, raise with the *minimum* legal raise (quantity_then_face ordering),
  preferring raises that are still reasonably likely under the same model.

Notes:
- This repo's engine supports an optional `ones_wild` rule. When enabled, bids on
  non-1 faces count both that face and 1s as matches. We reflect that in the
  per-die success probability.
- This is not a full Bayesian opponent-modeling agent (no learning of opponent
  tendencies). It's Bayesian in the sense of using a probabilistic belief over
  hidden dice.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, Optional

from . import register_agent
from .base import Agent
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid


def _per_die_match_prob(face: int, ones_wild: bool) -> float:
    """Probability a *single unknown die* counts as a match for `face`."""
    if ones_wild and face != 1:
        # matches if die is `face` or 1
        return 2.0 / 6.0
    return 1.0 / 6.0


def _count_known_matches(my_dice: Iterable[int], face: int, ones_wild: bool) -> int:
    """Count matches in known dice, mirroring core.rules.count_matches semantics."""
    my_dice = list(my_dice)
    count = sum(1 for d in my_dice if d == face)
    if ones_wild and face != 1:
        count += sum(1 for d in my_dice if d == 1)
    return count


def _binomial_tail_prob(n: int, p: float, k_min: int) -> float:
    """Compute P(X >= k_min) for X ~ Binomial(n, p).

    Implemented exactly via sums of combinations.
    Constraints in this project are small (<= 10-15 dice), so this is fast.
    """
    if k_min <= 0:
        return 1.0
    if k_min > n:
        return 0.0

    q = 1.0 - p
    total = 0.0
    for k in range(k_min, n + 1):
        total += comb(n, k) * (p**k) * (q ** (n - k))
    return total


@dataclass
class BayesianSettings:
    """Tunable knobs for the BayesianAgent."""

    # Base probability threshold below which we call liar.
    # 0.25 is a moderately aggressive default for 2-player, 5-dice each.
    base_call_threshold: float = 0.25

    # Some games get into long bid chains. As turn_index increases, we
    # *lower* the threshold a little so we don't call liar too early.
    threshold_decay_per_turn: float = 0.01

    # Hard clamps for numeric sanity.
    min_threshold: float = 0.05
    max_threshold: float = 0.60

    # When choosing a raise, require the candidate bid to have at least this
    # probability of being true. If none meet it, we still raise minimally.
    min_raise_truth_prob: float = 0.15


@register_agent("bayesian")
class BayesianAgent(Agent):
    """Probability-based agent that calls liar when a bid looks too unlikely."""

    def __init__(self, rng=None, settings: Optional[BayesianSettings] = None):
        super().__init__()
        self.rng = rng  # reserved for future (tie-breaks / etc.)
        self.settings = settings or BayesianSettings()

    def _total_dice(self, view) -> int:
        public = view["public"]
        if hasattr(public, "dice_counts") and public.dice_counts is not None:
            return int(sum(public.dice_counts))
        # Fallback, should rarely happen.
        config = view.get("config")
        if config is not None and getattr(config, "dice_distribution", None):
            return int(sum(config.dice_distribution))
        return int(getattr(config, "total_dice", 5) * getattr(config, "num_players", 2))

    def _truth_prob_for_bid(self, view, bid: Bid) -> float:
        config = view.get("config")
        ones_wild = bool(getattr(config, "ones_wild", False))

        my_dice = view["my_dice"]
        total = self._total_dice(view)
        unknown_n = max(0, total - len(my_dice))

        known = _count_known_matches(my_dice, bid.face, ones_wild)
        need_from_unknown = max(0, bid.quantity - known)
        p = _per_die_match_prob(bid.face, ones_wild)
        return _binomial_tail_prob(unknown_n, p, need_from_unknown)

    def _call_threshold(self, view) -> float:
        public = view["public"]
        turn_index = int(getattr(public, "turn_index", 0) or 0)

        thr = self.settings.base_call_threshold - (turn_index * self.settings.threshold_decay_per_turn)
        thr = max(self.settings.min_threshold, min(self.settings.max_threshold, thr))
        return thr

    def _iter_higher_bids(self, last_bid: Optional[Bid], config, total_dice: int):
        """Generate legal higher bids in increasing order (minimal raise first)."""
        faces = tuple(getattr(config, "faces", (1, 2, 3, 4, 5, 6)))

        # quantity_then_face ordering (matches Bid.is_higher_than in this repo)
        if last_bid is None:
            start_q = 1
            start_face_idx = 0
        else:
            start_q = last_bid.quantity
            start_face_idx = 0

        for q in range(start_q, total_dice + 1):
            for f in faces:
                candidate = Bid(q, f)
                if last_bid is None or candidate.is_higher_than(last_bid):
                    try:
                        candidate.validate(config)
                    except Exception:
                        continue
                    yield candidate

    def choose_action(self, view):
        public = view["public"]
        config = view.get("config")
        my_dice = view["my_dice"]
        last_bid = public.last_bid

        total = self._total_dice(view)

        # Opening move: place a small, high-probability bid based on our dice.
        if last_bid is None:
            ones_wild = bool(getattr(config, "ones_wild", False))
            # Prefer the face we already have the most of (incl. 1s when wild for non-1 bids).
            faces = tuple(getattr(config, "faces", (1, 2, 3, 4, 5, 6)))
            best_face = None
            best_count = -1
            for f in faces:
                c = _count_known_matches(my_dice, f, ones_wild)
                if c > best_count:
                    best_face, best_count = f, c
            # Make the bid conservative: at least 1.
            q = max(1, best_count)
            q = min(q, total)
            bid = Bid(q, int(best_face))
            return BidAction(bid)

        # Deterministic impossibility guard-rail.
        if self.call_liar_deterministic(my_dice, last_bid, total):
            return CallLiarAction()

        # Probabilistic call.
        p_true = self._truth_prob_for_bid(view, last_bid)
        if p_true < self._call_threshold(view):
            return CallLiarAction()

        # Otherwise: raise minimally, but try to keep it not-too-unlikely.
        min_good_prob = self.settings.min_raise_truth_prob
        best_fallback = None

        for candidate in self._iter_higher_bids(last_bid, config, total):
            if best_fallback is None:
                best_fallback = candidate  # minimal legal raise

            p_cand = self._truth_prob_for_bid(view, candidate)
            if p_cand >= min_good_prob:
                return BidAction(candidate)

        # If nothing clears min_good_prob, just take minimal legal raise; if no raise exists, call liar.
        if best_fallback is not None:
            return BidAction(best_fallback)
        return CallLiarAction()

