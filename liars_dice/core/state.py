from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .bid import Bid
from .config import GameConfig


@dataclass
class PlayerState:
    player_id: int
    num_dice: int
    private_dice: List[int] = field(default_factory=list)
    agent_id: Optional[str] = None


@dataclass
class PublicState:
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
    config: GameConfig
    players: Tuple[PlayerState, PlayerState]
    public: PublicState

