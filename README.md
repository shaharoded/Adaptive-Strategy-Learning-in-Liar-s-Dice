
# Liar's Dice - Adaptive Strategy Learning Engine

This repository implements a **headless, modular Liar's Dice game engine** designed for AI research, agent development, and strategy analysis. The engine supports:

- **2 players (agents)**, default 5 dice per round (configurable distribution)
- **Clean separation** between game logic, agents, UI, and persistence
- **Turn-by-turn event recording** for replay, analysis, and training
- **Plug-and-play agent interface** for rule-based, statistical, or learning agents

## Project Overview

The engine models the abstract game of Liar's Dice, allowing you to:
- Run self-play matches between any agents
- Collect detailed event logs and state transitions
- Experiment with new agent strategies, including RL/DRL
- Analyze and classify opponent strategies from collected data

### Architecture
- **Core logic**: Game rules, state transitions, bid validation, dice rolling
- **Agents**: Abstract agent interface (`Agent`), with pluggable implementations (see `liars_dice/agents/`)
- **Persistence**: Event recording and replay (see `liars_dice/persistence/`)
- **UI**: (Optional, not required for engine use)

## Requirements

- Python 3.8+
- No external dependencies required for core engine and basic agents

## Installation

Clone the repository and (optionally) create a virtual environment:

```powershell
git clone <repo-url>
cd Adaptive-Strategy-Learning-in-Liar-s-Dice
python -m venv .venv
.venv\Scripts\activate
```

## Running Tests

Run all unit tests from Root:

```powershell
python -m unittest discover -s tests -v
```

## How to Use the Engine

### 1. Create Agents
Implement new agents by subclassing `Agent` (see `liars_dice/agents/base.py`). Example: `RandomAgent` in `liars_dice/agents/random_agent.py`.


### 2. Running the Engine

You can use the engine in two main ways:

#### A. Running from Python (Agent-vs-Agent / Self-Play)
Use this mode to run automated matches between agents, collect data, or train AI models. This is ideal for research, self-play, and batch experiments.

**Example: Two agents playing against each other**

```python
from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine
from liars_dice.agents.random_agent import RandomAgent

config = GameConfig(dice_distribution=(2,3), ones_wild=False, rng_seed=42)
engine = GameEngine(config)
agent0 = RandomAgent()
agent1 = RandomAgent()

engine.start_new_round()
while not engine.is_terminal():
    current = engine.state.public.current_player
    agent = agent0 if current == 0 else agent1
    view = engine.get_view(current)
    action = agent.choose_action(view)
    engine.apply_action(current, action)
```

**Purpose:**
- Run self-play matches between any agents
- Collect event/state data for analysis or training
- Useful for developing/testing new agent strategies

#### B. Running from CLI (Human-vs-Agent)
Use this mode to play interactively as a human against an agent, using the provided CLI interface. This is ideal for manual testing, demos, or playing the game yourself.

**Example:**

```powershell
python -m UI.cli
```

You will be prompted for your moves in the terminal, while the agent plays automatically.

**Purpose:**
- Play as a human against a bot
- Useful for debugging, demonstrations, or exploring the game manually

---

### 4. Collecting Data

- **Events**: Every action, dice roll, and round outcome is recorded as an event (see `engine.get_events()` and `persistence/recorder.py`).
- **Turn Log**: Full state snapshots after each action are stored in `engine.turn_log`.
- **Persistence**: Use `InMemoryRecorder` or extend with file/DB recorders for long-term storage.

#### What is Collected?
- All dice rolls (per player, per round)
- Every bid and action (with player, bid details)
- Calls of "liar" and round outcomes (winner, loser, revealed dice)
- Full state transitions (for replay or training)

#### Where is it Collected?
- In-memory: `engine._events`, `engine.turn_log`, and via `InMemoryRecorder`
- Extendable: Implement custom recorders in `liars_dice/persistence/recorder.py`

## Extending the Project: AI & Agents
### Agent Development Roadmap
1. **Rule-based agents**: Implement agents with simple heuristics (e.g., always raise, always call liar after N turns, etc.)
2. **Statistical agents**: Use probability and opponent modeling
3. **Reinforcement Learning (RL/DRL) agents**: Train agents via self-play using collected event/state data
4. **Strategy classifier**: Build a model to classify opponent strategies and adapt play accordingly

See `liars_dice/agents/` for agent templates. All agents must implement `choose_action(view)`.

## Data for Training & Analysis

The engine is designed for **reproducible, event-sourced data collection**:
- Use event logs and turn logs for supervised or RL training
- Replay games deterministically using stored events
- Analyze agent behavior and strategy effectiveness

---

For more details, see `liars_dice_plan.md`.

