"""
llm_agent.py
An LLM-powered Liar's Dice agent that acts as an out-of-distribution adversary.
Not registered via @register_agent because it requires an injected API client
and cannot be instantiated with the standard no-arg constructor used by tournaments.
"""

import re
from typing import List

from .base import Agent
from ..core.bid import Bid
from ..core.actions import BidAction, CallLiarAction, Action


class LLMResponseError(Exception):
    """Raised when the LLM returns a response that cannot be parsed into a valid action."""
    pass


class LLMAdversaryAgent(Agent):
    """
    Purpose: Wraps an OpenAI-compatible chat client in the standard Agent interface.
             Intended as an out-of-distribution adversary for evaluating the AdaptiveAgent.
    Method:  Formats the player's partial view into a natural-language prompt, calls the
             LLM, and parses a BID / CALL response.  Raises on any API or parse failure.

    Args:
        client: Initialised OpenAI client (``openai.OpenAI`` instance).
        model (str): OpenAI model id, e.g. ``"gpt-4o-mini"``, ``"gpt-5"``, ``"o3"``.
        temperature (float): Sampling temperature (ignored for o1-o4 and gpt-5+ families).
        verbose (bool): Print each prompt/response pair when True.

    Memory: the agent accumulates a one-line summary of every completed round within the
    current game and prepends it to each prompt.  Call reset() between games to clear it.
    """

    SYSTEM_PROMPT = (
        "You are playing Liar's Dice. Rules:\n"
        "- Each player holds secret dice (standard 6-sided).\n"
        "- Players alternate making bids about the TOTAL dice across both players.\n"
        "  A bid is (quantity, face): \"I claim there are at least Q dice showing face F.\"\n"
        "- Each new bid must be strictly higher than the last: either a higher quantity,\n"
        "  or same quantity with a higher face value.\n"
        "- Instead of bidding you may call LIAR to challenge the last bid.\n"
        "  If the bid is false the bidder loses; otherwise the caller loses.\n"
        "- Ones are NOT wild. Valid faces: 1 2 3 4 5 6.\n\n"
        "Reply with EXACTLY one of:\n"
        "  BID <quantity> <face>   — e.g. BID 3 4\n"
        "  CALL                    — challenge the last bid\n\n"
        "Output only the action, nothing else."
    )

    def __init__(self, client, model: str = "gpt-4o-mini",
                 temperature: float = 0.7, verbose: bool = False):
        self._client        = client
        self._model         = model
        self._temperature   = temperature
        self._verbose       = verbose
        self._round_history: List[str] = []  # one compact line per completed round within the game
        self._total_calls   = 0
        # o1-o4 and gpt-5+ reject the temperature parameter entirely
        self._supports_temperature = bool(
            re.match(r"^gpt-4o", model) or re.match(r"^gpt-4\.1", model)
        )

    # ── inter-round memory ─────────────────────────────────────────────────────

    def record_round(self, bid_history: list, my_dice: int, opp_dice: int,
                     i_won: bool, was_true: bool) -> None:
        """
        Purpose: Append a one-line summary of the just-completed round to the memory buffer.
        Method:  Encodes the bid sequence, outcome, and die counts into a compact string;
                 trims the buffer to the last ``memory_rounds`` entries.
                 Call this from the game loop immediately after each round ends,
                 before start_new_round() is called.

        Args:
            bid_history (list[Bid]): Ordered list of Bid objects from the round.
            my_dice (int): Number of dice the LLM held at the start of this round.
            opp_dice (int): Number of dice the opponent held at the start of this round.
            i_won (bool): Whether the LLM won this round.
            was_true (bool): Whether the last bid was true when liar was called.
        """
        bids_str = " → ".join(f"{b.quantity}×{b.face}" for b in bid_history)
        outcome  = "TRUE bid, caller lost" if was_true else "FALSE bid, bidder lost"
        result   = "you won" if i_won else "you lost"
        self._round_history.append(f"[{my_dice}v{opp_dice}] {bids_str} | LIAR({outcome}) → {result}")

    # ── internal helpers ───────────────────────────────────────────────────────

    def _build_user_prompt(self, view: dict) -> str:
        """
        Purpose: Serialise the player view into a human-readable prompt.
        Method:  Prepends inter-round history (if any), then adds current-round state
                 and a hint about the minimum valid next bid.

        Args:
            view (dict): Engine view dict for this player.

        Returns:
            str: Formatted prompt string.
        """
        pub         = view["public"]
        my_dice     = sorted(view["my_dice"])
        my_id       = view["player_id"]
        opp_id      = 1 - my_id
        dice_counts = pub.dice_counts
        total_dice  = dice_counts[0] + dice_counts[1]

        last_bid_str = (
            "None (you bid first)" if pub.last_bid is None
            else f"{pub.last_bid.quantity} × face {pub.last_bid.face}"
        )

        bid_lines = "\n".join(
            f"  {i+1}. {b.quantity} × face {b.face}"
            for i, b in enumerate(pub.bid_history)
        ) or "  (no bids yet)"

        # Inter-round context block (empty string when memory is off)
        if self._round_history:
            history_block = (
                "=== MATCH HISTORY (last rounds) ===\n"
                + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self._round_history))
                + "\n\n"
            )
        else:
            history_block = ""

        return (
            f"{history_block}"
            f"=== YOUR TURN ===\n"
            f"Your dice ({dice_counts[my_id]} dice): {my_dice}\n"
            f"Opponent dice count: {dice_counts[opp_id]} (faces hidden)\n"
            f"Total dice in play: {total_dice}\n"
            f"\nBid history this round:\n{bid_lines}\n"
            f"\nLast bid: {last_bid_str}\n"
            f"\nYour action:"
        )

    def _parse_response(self, text: str, view: dict) -> Action:
        """
        Purpose: Parse the LLM text response into a valid game Action.
        Method:  Regex match for BID q f or CALL; validates bid legality against the view.
                 Raises LLMResponseError if the response is unparseable or illegal.

        Args:
            text (str): Raw LLM response string.
            view (dict): Current game view (for validation).

        Returns:
            Action: A valid BidAction or CallLiarAction.

        Raises:
            LLMResponseError: If the response cannot be parsed or produces an illegal move.
        """
        normalised = text.strip().upper()

        if re.match(r"^CALL", normalised):
            if view["public"].last_bid is None:
                raise LLMResponseError(
                    f"LLM returned CALL but there is no bid to challenge. Response: {text!r}"
                )
            return CallLiarAction()

        m = re.search(r"BID\s+(\d+)\s+(\d+)", normalised)
        if m:
            q, f   = int(m.group(1)), int(m.group(2))
            config = view["config"]
            total  = sum(view["public"].dice_counts)

            if f not in config.faces:
                raise LLMResponseError(
                    f"LLM bid face {f} is not a valid face {config.faces}. Response: {text!r}"
                )
            if not (1 <= q <= total):
                raise LLMResponseError(
                    f"LLM bid quantity {q} out of range [1, {total}]. Response: {text!r}"
                )

            bid = Bid(quantity=q, face=f)
            if not bid.is_higher_than(view["public"].last_bid):
                raise LLMResponseError(
                    f"LLM bid ({q},{f}) does not beat last bid "
                    f"{view['public'].last_bid}. Response: {text!r}"
                )

            return BidAction(bid)

        raise LLMResponseError(
            f"LLM response could not be parsed as BID or CALL. Response: {text!r}"
        )

    # ── public interface ───────────────────────────────────────────────────────

    def choose_action(self, view: dict) -> Action:
        """
        Purpose: Select the next game action via the LLM.
        Method:  Build prompt → call OpenAI → parse response.
                 Raises on API failure or unparseable/illegal response.

        Args:
            view (dict): Player view from GameEngine.get_view().

        Returns:
            Action: BidAction or CallLiarAction.

        Raises:
            Exception: Re-raises any OpenAI API exception.
            LLMResponseError: If the response cannot be parsed into a valid action.
        """
        self._total_calls += 1
        user_prompt = self._build_user_prompt(view)

        api_params = {
            "model":    self._model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        }
        if self._supports_temperature:
            api_params["temperature"] = self._temperature

        resp = self._client.chat.completions.create(**api_params)
        raw  = resp.choices[0].message.content or ""

        if self._verbose:
            print(f"[LLMAdversaryAgent] prompt:\n{user_prompt}")
            print(f"[LLMAdversaryAgent] response: {raw!r}")

        return self._parse_response(raw, view)

    def reset(self):
        """Purpose: Clear all per-game state including inter-round memory."""
        self._total_calls   = 0
        self._round_history = []
