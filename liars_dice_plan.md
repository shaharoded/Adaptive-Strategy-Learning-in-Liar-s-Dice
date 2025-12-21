# Liar’s Dice (2 Agents, 5 Total Dice) — Python Game Engine Plan

This repository will contain a **headless (no UI)** Liar’s Dice engine designed for:
- **2 players (agents)**.
- **10 total dice per round** (distributed across players).
- Clean separation between **game logic**, **agents**, **UI**, and **persistence**.
- Full **turn-by-turn recording** to support later DB storage and replay.

The goal is to implement the *abstract game* once, then plug in many different agents (bots), and later add UI + DB without touching core rules.

---

## Assumed rule-set (configurable)

Because “Liar’s Dice” has several variants, the engine will implement a **configurable rule set** with safe defaults.

### Default rules (recommended for this project)
1. **Each round**:
   - Both players roll their private dice.
   - Players alternate turns making **bids** or calling **liar**.
2. **A bid** is: `(quantity, face)` meaning “there are at least `quantity` dice showing `face` among all dice”.
3. **Bid progression constraint**:
   - Each new bid must be **strictly higher** than the previous bid under a deterministic ordering.
   - Default ordering: higher quantity wins; if quantity equal, higher face wins.
4. **Call Liar**:
   - The caller claims the previous bid is false.
   - Reveal all dice and count matches.
   - If count >= quantity → the bid was true → **caller loses**.
   - Else → bid was false → **previous bidder loses**.
5. **Winning**:
   - Since total dice are fixed to 5, the simplest version is **single-round**: one call ends the round and produces a winner.
   - The engine will still support “match play” (best-of-N rounds) as an optional wrapper later.

### Config toggles (supported by `GameConfig`)
- `ones_wild: bool` — if `True`, ones count as any face except possibly when bidding ones (configurable).
- `dice_distribution: tuple[int,int]` — default `(2,3)` (Player0 has 2 dice, Player1 has 3).
- `bid_ordering: "quantity_then_face" | "liars_dice_standard"` — default `quantity_then_face`.
- `allow_opening_bid_constraints` — e.g., minimum quantity, etc.
- `max_turns` — prevent infinite loops (shouldn’t happen with strict bid ordering, but useful defensively).

---

## High-level architecture

### Layers
1. **Core domain (pure logic)**
   - Dice, bids, legal moves, state transitions, winner detection.
   - Absolutely no UI/DB code.
2. **Agents**
   - An abstract interface (`Agent`) that returns an action given a **player-specific view** of the game state.
3. **Observers + Persistence**
   - An event stream of `GameEvent` objects (event sourcing).
   - Optional snapshotting of `GameState` for fast replay.
4. **Adapters**
   - UI adapter: subscribe to events + request actions.
   - DB adapter: store events/snapshots.

### Guiding principles
- **Determinism:** seedable RNG; record dice outcomes as events.
- **Replayability:** the engine can replay a game from stored events.
- **Strict legality:** invalid actions are rejected with clear errors.
- **Separation of concerns:** agents never mutate core state directly.

---

## Suggested repository structure

```
liars_dice/
  __init__.py

  core/
    config.py          # GameConfig
    dice.py            # Die, DiceCup
    bid.py             # Bid model + ordering
    actions.py         # BidAction, CallLiarAction
    state.py           # PlayerState, PublicState, GameState
    engine.py          # GameEngine (state machine)
    rules.py           # Helpers for counting, validation, wild-ones logic

  agents/
    base.py            # Agent ABC + views
    random_agent.py    # Example simple agent
    human_agent.py     # Placeholder for UI-driven actions later

  persistence/
    events.py          # GameEvent types (RollEvent, BidEvent, CallEvent, RevealEvent, EndEvent)
    recorder.py        # InMemoryRecorder / JSONLinesRecorder skeleton
    serializer.py      # serialize/deserialize events + snapshots

  tests/
    test_rules.py
    test_engine_flow.py
    test_replay.py

README.md
pyproject.toml (optional)
```

---

## Core game concepts (data models)

### `GameConfig`
**Purpose:** Centralizes all rule options and numeric constraints.

Key fields:
- `num_players = 2`
- `total_dice = 5`
- `dice_distribution = (2,3)` (must sum to 5)
- `faces = (1..6)`
- `ones_wild = False` (default)
- `bid_ordering = "quantity_then_face"`
- `max_turns = 64`
- `rng_seed: int | None` (optional)

### `Die` / `DiceCup`
**Purpose:** Rolling and representing dice.

- `Die.roll(rng) -> int`
- `DiceCup.roll(n: int, rng) -> list[int]`
- Dice results are stored in `PlayerState.private_dice`.

### `Bid`
**Purpose:** Represents a bid and provides strict ordering.

Fields:
- `quantity: int` (>=1)
- `face: int` (1..6)

Methods:
- `is_higher_than(other: Bid, config: GameConfig) -> bool`
- `validate(config)`

Ordering policy is centralized so every rule change affects legality consistently.

### Actions
All moves are represented explicitly.

- `Action` (base type)
  - `BidAction(bid: Bid)`
  - `CallLiarAction()`

This makes it easy to store actions as events and replay.

---

## State representation

### `PlayerState`
Private to the engine; partially visible to the owning player via a view.

Fields:
- `player_id: int`
- `num_dice: int`
- `private_dice: list[int]` (only known to that player + engine)
- `agent_id: str | None` (useful for experiment metadata)

### `PublicState`
Visible to everyone.

Fields:
- `round_index: int`
- `turn_index: int`
- `current_player: int`
- `last_bid: Bid | None`
- `bid_history: list[Bid]` (or derived from events)
- `status: "BIDDING" | "REVEAL" | "ENDED"`
- `winner: int | None`
- `loser: int | None`

### `GameState`
Composition:
- `config: GameConfig`
- `players: tuple[PlayerState, PlayerState]`
- `public: PublicState`

Design notes:
- Keep `GameState` as **plain data** (dataclasses).
- The engine is the only component allowed to mutate it (or return copies).

---

## Engine design (state machine)

### `GameEngine`
**Responsibility:** Apply actions, enforce legality, emit events.

Key methods:
- `start_new_round() -> None`
  - Rolls dice for each player according to distribution.
  - Emits `RoundStartedEvent` and `DiceRolledEvent` per player (dice values recorded!).
- `get_view(player_id: int) -> GameView`
  - Returns a **player-specific projection** containing:
    - public info + that player's private dice
    - legal actions (optional helper)
- `apply_action(player_id: int, action: Action) -> None`
  - Validates:
    - game not ended
    - correct player’s turn
    - action legal relative to `last_bid`
  - Updates state and emits events.
- `reveal_and_resolve(caller_id: int) -> None`
  - Counts matches, determines loser/winner, emits reveal + end events.
- `is_terminal() -> bool`

### Turn / flow
1. `start_new_round()`
2. Player `current_player` does one of:
   - `BidAction(new_bid)` → update `last_bid`, push history, advance turn
   - `CallLiarAction()` → reveal all dice, resolve winner, end round

### Bid legality rules
- If `last_bid is None`: any valid bid allowed.
- Else: `new_bid.is_higher_than(last_bid)` must be true.
- Quantity must be <= `total_dice` (configurable to allow impossible bids for bluffing; default disallow impossible bids to keep learning simpler).
  - **Recommendation:** allow any quantity 1..total_dice (so still bounded).
- Face must be within 1..6.

### Counting rule
- `count(face)` across both players’ dice.
- If `ones_wild`:
  - for face != 1: `count(face) + count(1)`
  - for face == 1: `count(1)` (or configurable alternative)

---

## Agent interface (for later bot players)

### `Agent` (abstract base class)
Agents should be able to:
- receive a view of the state,
- choose an action,
- optionally be notified of events.

Proposed minimal interface:
- `choose_action(view: GameView) -> Action`

Optional hooks:
- `on_game_start(config, player_id)`
- `on_event(event: GameEvent)`
- `on_game_end(result)`

### `GameView`
A player-specific, immutable view:
- `player_id`
- `public: PublicState`
- `my_dice: tuple[int, ...]`
- `legal_actions: tuple[Action, ...]` (optional convenience; can be computed)

**Important:** Agents must *not* access opponent dice until reveal events occur.

---

## Events & persistence plan (DB-ready)

The engine will emit a canonical stream of `GameEvent`s. This is the primary integration point for DB storage and UI updates.

### Why event sourcing
- Perfect for “save turns”
- Allows deterministic replay
- Easy to attach observers without coupling

### Event types (examples)
All events share:
- `game_id: str`
- `round_index: int`
- `turn_index: int`
- `timestamp_utc: str` (or injected clock)
- `event_type: str`
- `payload: dict`
- `player_type: str | None` — agent class name or 'Human' for the player who took the action (optional, always present for moves/actions; may be None for system events)

Concrete events:
- `GameCreatedEvent(config=...)`
- `RoundStartedEvent(dice_distribution=...)`
- `DiceRolledEvent(player_id, dice=[...], seed_info=..., player_type=...)`
- `BidPlacedEvent(player_id, bid={quantity,face}, player_type=...)`
- `LiarCalledEvent(player_id=caller_id, player_type=...)`
- `DiceRevealedEvent(all_dice={player0:[...], player1:[...]})`
- `RoundEndedEvent(winner_id, loser_id, last_bid, match_count, was_true)`
- (Optional) `InvalidActionEvent(...)` (generally better as exception, not event)

> **Note:** `player_type` is always included for any event representing a player action (bid, call, roll, etc.), and records the agent class or 'Human'. This enables downstream analysis and model training that conditions on or infers player strategy.

### Recorder interface
`GameRecorder`:
- `record(event: GameEvent) -> None`
- `flush() -> None` (for file/DB)
- `events() -> list[GameEvent]` (for in-memory / testing)

Implementations (planned):
- `InMemoryRecorder` for tests
- `JSONLinesRecorder(path)` for local logging
- DB adapter later: `SQLRecorder` / `MongoRecorder`, etc.

### Snapshots (optional)
- Periodic `GameStateSnapshot` for fast loading.
- Not required for 5-dice games, but included for extensibility.

---

## UI integration plan (later)

UI should not call core internals; it should:
1. Subscribe to events from a recorder/observer.
2. Render public state updates.
3. For human play, provide a `HumanAgent` that:
   - blocks/waits for UI input externally,
   - returns an `Action` to the engine when input arrives.

For non-blocking systems:
- Use an adapter layer that steps the engine:
  - `engine.step()` asks the current agent for an action and applies it.

---

## Error handling & validation strategy

- Illegal actions raise `IllegalMoveError` with structured details:
  - expected player, current bid, proposed bid, etc.
- Engine should never enter an inconsistent state:
  - every transition is validated
  - all state changes are centralized in `apply_action`

---

## Testing plan

1. **Bid ordering tests**
   - ensure strict monotonicity, edge cases
2. **Counting rule tests**
   - with and without wild ones
3. **Full round simulation**
   - deterministic via seed
4. **Replay tests**
   - run a game, record events, replay and compare final state

---

## Milestones (implementation order)

1. Implement domain models: `GameConfig`, `Bid`, `Action`
2. Implement dice roll + state initialization
3. Implement `GameEngine.apply_action` + legality checks
4. Implement reveal & resolution logic
5. Implement event types + `InMemoryRecorder`
6. Add `GameView` + `Agent` ABC + one simple agent
7. Add JSON-lines recorder + replay utility
8. Add tests

---

## Notes on the “5 total dice” constraint

Default distribution will be **(2,3)** for two players. This makes hidden information asymmetric (interesting for agents).
If you prefer symmetric play, we can instead:
- keep both players at 2 dice and add 1 “table die” (public) or “shared hidden die” (engine-only),
but that’s a different game. The engine will be designed so distribution is configurable.

---

## Next step

After you approve this plan, the next deliverable will be the initial implementation skeleton with:
- `GameEngine` + core models
- event recorder
- minimal agent interface
- deterministic tests
