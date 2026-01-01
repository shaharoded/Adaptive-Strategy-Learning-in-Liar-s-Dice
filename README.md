
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
- torch (optional)
- tensorboard (optional)

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

## Available Scripts

The project includes several scripts for different simulation and testing scenarios:

### 1. `run_experiments.py` - Single-Round Games
Simulates individual **single-round games** between two agents. Each game ends when someone calls "Liar".

**Use case**: Quick testing, data collection, analyzing single-round strategies

**Run from project root**:
```powershell
python -m scripts.run_experiments
```

**Configuration**: Edit the script's `main()` function to set:
- `agent_1` and `agent_2`: Agent names (e.g., "random", "heuristic", "nash_cfr")
- `number_of_games`: How many games to run
- `data_dir`: Output directory for CSV files (default: "data")

**Output**: 
- `data/game_summary.csv`: Game results and statistics
- `data/game_trajectory.csv`: Turn-by-turn events and states

### 2. `full_game.py` - Full Multi-Round Matches
Simulates **complete matches** with multiple rounds. After each round, the loser loses 1 die. Match continues until one player reaches 0 dice.

**Use case**: Realistic gameplay simulation, testing endurance/adaptation strategies

**Run from project root**:
```powershell
python -m scripts.full_game
```

**Configuration**: Edit the script's `main()` function to set:
- `agent_1` and `agent_2`: Agent names
- `number_of_matches`: How many full matches to run
- `data_dir`: Output directory (default: "data")

**Output**:
- `data/match_summary.csv`: Match results with rounds played
- `data/match_trajectory.csv`: All events across all rounds

### 3. `run_tournament.py` - Single-Round Tournament
Runs a **round-robin tournament** where all agent pairings play single-round games against each other.

**Use case**: Comparing agent performance across single rounds

**Run from project root**:
```powershell
python -m scripts.run_tournament --agents all --games 10 --data-dir data
```

**Options**:
- `--agents`: Comma-separated agent list or "all" (e.g., "random,heuristic" or "all")
- `--games`: Number of games per ordered pairing
- `--data-dir`: Output directory

**Output**:
- `data/game_summary.csv`: Individual game results
- `data/game_trajectory.csv`: Turn-by-turn events
- `data/tournament_summary.csv`: Per-pairing statistics
- `data/agent_stats.csv`: Aggregated win rates per agent
- `data/win_percentages.png`: Win percentage chart

### 4. `run_tournament_full_game.py` - Full-Match Tournament
Same as tournament #3, but uses **full multi-round matches** instead of single rounds.

**Use case**: Comprehensive agent comparison with realistic match conditions

**Run from project root**:
```powershell
python -m scripts.run_tournament_full_game --agents all --games 10 --data-dir data
```

**Options**: Same as `run_tournament.py`

**Output**: Same files as `run_tournament.py` but with match-level data

## User Interfaces

The project includes both CLI and GUI interfaces for human play:

### CLI (Command-Line Interface)

Play Liar's Dice in the terminal against an agent.

**Run from project root**:
```powershell
python -m UI.cli
```

**Features**:
- Interactive text-based gameplay
- Choose from available agents
- View game rules
- All games are saved to `data/game_summary.csv` and `data/game_trajectory.csv`

**Controls**:
- Menu navigation: Enter 1, 2, or 3
- Actions: Choose "Bid" or "Call Liar"
- Input bids: Enter quantity and face values when prompted

### GUI (Graphical Interface)

Play Liar's Dice with a visual interface.

**Run from project root**:
```powershell
python -m UI.gui
```

**Features**:
- Visual dice display (your dice shown, opponent's hidden)
- Interactive buttons for bidding and calling liar
- Bid history tracking
- Real-time game status updates
- All games are saved to `data/game_summary.csv` and `data/game_trajectory.csv`

**Controls**:
- Select opponent agent from dropdown
- Click "Start Game" to begin
- Enter quantity and face in text boxes
- Click "Bid" or "Call Liar" buttons
- Click "New Round" to play again

**Note**: Both CLI and GUI record game data to the same CSV files, allowing you to analyze human gameplay alongside agent self-play data.

## Training NashCFRAgent Policies (CFR)

To train NashCFRAgent policies for all dice count combinations (multi-policy CFR):

1. Install requirements (Python 3.8+, torch, tensorboard recommended for live monitoring):
    ```powershell
    pip install torch tensorboard
    ```

2. Run the training script from the project root:
    ```powershell
    python -m scripts.train_nash_cfr_agent --num_players 2 --max_dice 5 --iterations 10000 --tensorboard runs/nash_cfr --checkpoint liars_dice/agents/weights/nash_cfr_policy.pkl
    ```
    - This will train CFR policies for all dice count combinations (2 players, 1–5 dice each) and save to liars_dice/agents/weights/nash_cfr_policy.pkl.
    - Use `--tensorboard <logdir>` to log average regret for live monitoring.
    - Use `--checkpoint <path>` to resume training from a previous checkpoint.

3. To view training progress live:
    ```powershell
    tensorboard --logdir runs/nash_cfr
    ```
    - Open the provided URL in your browser to see regret curves and training status.

4. The trained policy file will be loaded automatically by NashCFRAgent when used in the engine.

---
## How to Use the Engine

### 1. Create Agents
Implement new agents by subclassing `Agent` (see `liars_dice/agents/base.py`). Example: `RandomAgent` in `liars_dice/agents/random_agent.py`.


### 2. Running the Engine

You can use the engine in two main ways:

#### A. Running from Python (Agent-vs-Agent / Self-Play)
Use this mode to run automated matches between agents, collect data, or train AI models. This is ideal for research, self-play, and batch experiments.

**Option 1: Two agents playing against each other (single game)**

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

**Option 2: Batch self-play experiments (multiple games, data collection)**

You can run a batch of games and collect results using the provided experiment script (from the project's root):

```powershell
python -m scripts.run_experiments
```

Edit the configuration section at the top of scripts/run_experiments.py to set the number of games, agent types, and output directory. Results and statistics will be saved to the specified directory (default: ../results).

**Purpose:**
- Run self-play matches between any agents (single or batch)
- Collect event/state data for analysis or training
- Useful for developing/testing new agent strategies

#### B. Running the User Interfaces (Human-vs-Agent)
Use the CLI or GUI to play interactively as a human against an agent. This is ideal for manual testing, demos, or playing the game yourself.

**CLI Example:**
```powershell
python -m UI.cli
```

**GUI Example:**
```powershell
python -m UI.gui
```

**Purpose:**
- Play as a human against a bot
- Useful for debugging, demonstrations, or exploring the game manually
- Both interfaces save game data to CSV files for later analysis

For detailed UI usage, see the **User Interfaces** section above.

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

## TO-DO:

1. GUI and CLI should play a full game, not a single round (with decreasing number of dice and no option to change agent unless starting a new game).
2. Data collection method should be tuned for training.
---

For more details, see `liars_dice_plan.md`.

