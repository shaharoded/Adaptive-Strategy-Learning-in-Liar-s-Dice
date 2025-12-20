
"""
state.py
Defines all game state dataclasses for Liar's Dice: PlayerState, PublicState, GameState.
Related modules:
- engine.py: Mutates and reads GameState during play.
- bid.py: Used in bid history and last_bid.
- config.py: GameConfig is part of GameState.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .bid import Bid
from .config import GameConfig


@dataclass
class PlayerState:
    """
    Stores private state for a single player.
    Fields:
        player_id (int): Player index.
        num_dice (int): Number of dice held.
        private_dice (list[int]): Player's dice (hidden from opponent).
        agent_id (str|None): Optional agent identifier.
    """
    player_id: int
    num_dice: int
    private_dice: List[int] = field(default_factory=list)
    agent_id: Optional[str] = None


@dataclass
class PublicState:
    """
    Stores public state visible to all players and agents.
    Fields:
        round_index (int): Current round number.
        turn_index (int): Current turn number.
        current_player (int): Player whose turn it is.
        last_bid (Bid|None): Most recent bid.
        bid_history (list[Bid]): All bids this round.
        status (str): Game status (NOT_STARTED, BIDDING, REVEAL, ENDED).
        winner (int|None): Winner of the round.
        loser (int|None): Loser of the round.
    """
    round_index: int = 0
    turn_index: int = 0
    current_player: int = 0
    last_bid: Optional[Bid] = None
    bid_history: List[Bid] = field(default_factory=list)
    status: str = "NOT_STARTED"  # BIDDING | REVEAL | ENDED
    winner: Optional[int] = None
    loser: Optional[int] = None


@dataclass
class GameState:
    """
    Composite state for the entire game: config, both players, and public state.
    Fields:
        config (GameConfig): Game configuration.
        players (tuple): Tuple of PlayerState for each player.
        public (PublicState): Public game state.
    """
    config: GameConfig
    players: Tuple[PlayerState, PlayerState]
    public: PublicState

