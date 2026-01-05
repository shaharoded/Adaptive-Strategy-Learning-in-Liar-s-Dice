import sys
from typing import Optional


from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.actions import BidAction, CallLiarAction, Action
from liars_dice.core.bid import Bid
from liars_dice.core.reward import get_reward
from liars_dice.agents.base import Agent
from liars_dice.agents import AGENT_MAP
from liars_dice.persistence import csv_io

import os
import datetime
import hashlib

class HumanAgent(Agent):
    """
    Placeholder agent for human input in CLI mode.
    The choose_action method is not used; input is handled directly in the CLI loop.
    """
    def __init__(self):
        pass

    def choose_action(self, view):
        """
        Not implemented. Human input is handled in the CLI loop, not via this method.
        """
        raise NotImplementedError("HumanAgent.choose_action should not be called")



def choose_agent(name: str) -> Agent:
    """
    Return an Agent instance by name.
    Args:
        name (str): Name of the agent (e.g., 'random').
    Returns:
        Agent: The corresponding agent instance.
    Raises:
        ValueError: If the agent name is unknown.
    """
    name = name.lower()
    if name in AGENT_MAP:
        return AGENT_MAP[name]()
    raise ValueError(f"Unknown agent: {name}")



def print_state(view):
    """
    Print the current public state and player's dice to the terminal.
    Args:
        view (dict): Player-specific view from engine.get_view().
    """
    public = view["public"]
    my_dice = view["my_dice"]
    print("\n=== ROUND {round} ===".format(round=public.round_index))
    print(f"Your dice: {tuple(my_dice)}")
    last = public.last_bid
    if last is None:
        print("No bids yet.")
    else:
        print(f"Last bid: quantity: {last.quantity}, face: {last.face}")
    print(f"Current player: {public.current_player}")
    print(f"Turn index: {public.turn_index}")



def prompt_action(view) -> Optional[Action]:
    """
    Prompt the human player for an action (Bid or Call Liar).
    Args:
        view (dict): Player-specific view from engine.get_view().
    Returns:
        Action or None: The chosen action, or None if input is invalid.
    """
    public = view["public"]
    last = public.last_bid
    # Present options: Bid or Call Liar
    print("\nChoose action:")
    print("  1) Bid")
    print("  2) Call Liar")
    choice = input("Enter choice (1-2): ").strip()
    if choice == "2":
        return CallLiarAction()
    elif choice == "1":
        # Prompt for quantity and face
        while True:
            qty_s = input("Enter quantity (int): ").strip()
            try:
                qty = int(qty_s)
            except ValueError:
                print("Please enter a valid integer for quantity.")
                continue
            face_s = input("Enter face (1-6): ").strip()
            try:
                face = int(face_s)
            except ValueError:
                print("Please enter a valid integer for face.")
                continue
            try:
                bid = Bid(qty, face)
                bid.validate(view.get("config"))
            except Exception as e:
                print(f"Invalid bid: {e}")
                continue
            # If last exists, ensure bid is higher
            if last is not None and not bid.is_higher_than(last):
                print("Bid must be higher than the last bid.")
                continue
            return BidAction(bid)
    else:
        print("Choice not recognized.")
        return None



def show_rules(config: GameConfig):
    """
    Print the current game rules and configuration to the terminal.
    Args:
        config (GameConfig): The game configuration.
    """
    print("\n=== GAME RULES ===")
    print(f"Players: {config.num_players}")
    # display dice distribution if present
    if config.dice_distribution:
        print(f"Dice distribution: {config.dice_distribution} (sum {sum(config.dice_distribution)})")
    else:
        print(f"Total dice per player: {config.total_dice}")
    print(f"Faces: {tuple(config.faces)}")
    print(f"Ones wild: {config.ones_wild}")
    print(f"Bid ordering: {config.bid_ordering}")



def play_against(agent_name: str = "random", config: Optional[GameConfig] = None):
    """
    Play a full match of Liar's Dice as a human (player 0) against an agent (player 1) in the CLI.
    The match consists of multiple rounds where the loser of each round loses one die.
    The match ends when one player has zero dice.
    Args:
        agent_name (str): Name of the agent to play against (default 'random').
        config (GameConfig, optional): Game configuration. If None, uses default config.
    """
    # Setup game: create config at runtime so rng_seed can be None (non-deterministic)
    if config is None:
        config = GameConfig(dice_distribution=(5, 5), rng_seed=None)
    engine = GameEngine(config)
    agent = choose_agent(agent_name)

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    trajectory_csv = os.path.join(data_dir, "game_trajectory.csv")
    summary_csv = os.path.join(data_dir, "game_summary.csv")
    trajectory_header = csv_io.get_trajectory_header()
    summary_header = csv_io.get_summary_header()

    # Let human be player 0 and agent be player 1
    human_id = 0
    agent_id = 1
    
    # Full match loop: play rounds until one player has zero dice
    # Each round is saved as a separate game
    print("\n=== STARTING FULL MATCH ===")
    print(f"Playing against: {type(agent).__name__}")
    print(f"Starting dice: 5 each\n")
    
    round_number = 0
    while True:
        # Check if match is over (any player has zero dice)
        p0, p1 = engine.state.players
        if p0.num_dice <= 0 or p1.num_dice <= 0:
            break
            
        # Start new round - each round gets its own game_id
        round_number += 1
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        raw_id = f"cli_{timestamp}_{os.getpid()}_{agent_name}_round{round_number}"
        game_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        trajectory_rows = []
        
        def record_event(event_type, payload, player_type=None, turn_index=None, player=None, state=None, action=None, reward_val=None):
            r = reward_val if reward_val is not None else get_reward(event_type, state, action, player, engine.state.public)
            trajectory_rows.append({
                "game_id": game_id,
                "event_type": event_type,
                "turn_index": turn_index if turn_index is not None else engine.state.public.turn_index,
                "player": player,
                "player_type": player_type,
                "payload": str(payload),
                "timestamp": timestamp,
                "state": str(state) if state is not None else "",
                "action": str(action) if action is not None else "",
                "reward": r,
            })
        
        engine.start_new_round()
        
        print(f"\n{'='*50}")
        print(f"ROUND {round_number}")
        print(f"{'='*50}")
        print(f"Your dice: {p0.num_dice} | Agent dice: {p1.num_dice}\n")
        
        record_event("RoundStarted", {"round": engine.state.public.round_index}, player_type=None, turn_index=engine.state.public.turn_index, player=None, state=engine.get_view(0), action=None, reward_val=0)
        p0, p1 = engine.state.players
        record_event("DiceRolled", {"player0": p0.private_dice.copy(), "player1": p1.private_dice.copy()}, player_type=None, turn_index=engine.state.public.turn_index, player=None, state=engine.get_view(0), action=None, reward_val=0)

        round_steps = 0
        round_bids = 0
        round_calls = 0
        while not engine.is_terminal():
            # Show human view only when it's their turn; otherwise ask agent for action and apply
            current = engine.state.public.current_player
            if current == human_id:
                view = engine.get_view(human_id)
                print_state(view)
                action = None
                while action is None:
                    action = prompt_action(view)
                try:
                    engine.apply_action(human_id, action)
                    round_steps += 1
                    # Record human action
                    if isinstance(action, BidAction):
                        round_bids += 1
                        record_event("BidPlaced", {"player": human_id, "bid": (action.bid.quantity, action.bid.face)}, player_type="Human", turn_index=engine.state.public.turn_index, player=human_id, state=view, action=action, reward_val=0)
                    elif isinstance(action, CallLiarAction):
                        round_calls += 1
                        record_event("LiarCalled", {"caller": human_id, "last_bid": engine.state.public.last_bid}, player_type="Human", turn_index=engine.state.public.turn_index, player=human_id, state=view, action=action, reward_val=0)
                except IllegalMoveError as e:
                    print(f"Illegal move: {e}")
                    continue
            else:
                # Agent's turn
                view = engine.get_view(agent_id)
                action = agent.choose_action(view)
                print(f"Agent action: {type(action).__name__}")
                try:
                    engine.apply_action(agent_id, action)
                    round_steps += 1
                    # Record agent action
                    if isinstance(action, BidAction):
                        round_bids += 1
                        record_event("BidPlaced", {"player": agent_id, "bid": (action.bid.quantity, action.bid.face)}, player_type=type(agent).__name__, turn_index=engine.state.public.turn_index, player=agent_id, state=view, action=action, reward_val=0)
                    elif isinstance(action, CallLiarAction):
                        round_calls += 1
                        record_event("LiarCalled", {"caller": agent_id, "last_bid": engine.state.public.last_bid}, player_type=type(agent).__name__, turn_index=engine.state.public.turn_index, player=agent_id, state=view, action=action, reward_val=0)
                except IllegalMoveError as e:
                    # If agent made illegal move, treat as pass/call liar
                    print(f"Agent made illegal move: {e}. Agent will call liar instead.")
                    engine.apply_action(agent_id, CallLiarAction())
                    round_steps += 1
                    round_calls += 1
                    record_event("LiarCalled", {"caller": agent_id, "last_bid": engine.state.public.last_bid}, player_type=type(agent).__name__, turn_index=engine.state.public.turn_index, player=agent_id, state=view, action="CallLiarAction", reward_val=0)

        # Round ended; show results
        public = engine.state.public
        print("\n--- ROUND ENDED ---")
        print(f"Winner: Player {public.winner}")
        print(f"Loser: Player {public.loser}")
        print(f"Final bid: bid quantity:{public.last_bid.quantity}, face:{public.last_bid.face}")
        # Reveal dice
        p0, p1 = engine.state.players
        print(f"Player 0 dice: {p0.private_dice}")
        print(f"Player 1 dice: {p1.private_dice}")

        # Record round end and dice reveal
        record_event("DiceRevealed", {"all_dice": {0: p0.private_dice, 1: p1.private_dice}}, player_type=None, turn_index=engine.state.public.turn_index, player=None, state=engine.get_view(0), action=None, reward_val=0)
        # Assign reward at end of round using get_reward
        final_state = engine.get_view(0)
        final_action = None
        final_reward = get_reward("RoundEnded", final_state, final_action, None, public)
        record_event("RoundEnded", {"winner": public.winner, "loser": public.loser, "match_count": None, "was_true": None}, player_type=None, turn_index=engine.state.public.turn_index, player=None, state=final_state, action=final_action, reward_val=final_reward)
        
        # Save this round as a separate game
        csv_io.append_rows_to_csv(trajectory_rows, trajectory_csv, trajectory_header)
        
        summary_row = {
            "game_id": game_id,
            "game_index": round_number - 1,
            "timestamp": timestamp,
            "agent0": "Human",
            "agent1": type(agent).__name__,
            "winner": public.winner,
            "loser": public.loser,
            "steps": round_steps,
            "bids": round_bids,
            "calls": round_calls,
            "bluffs_called": 0,
            "error": None,
            "end_reason": "winner declared" if public.winner is not None else None,
        }
        csv_io.append_row_to_csv(summary_row, summary_csv, summary_header)
        
        # Decrement loser's dice
        if public.loser is not None:
            loser = public.loser
            engine.state.players[loser].num_dice = max(0, engine.state.players[loser].num_dice - 1)
            print(f"\nPlayer {loser} loses a die! Now has {engine.state.players[loser].num_dice} dice.")
    
    # Match ended - determine winner
    p0, p1 = engine.state.players
    if p0.num_dice <= 0:
        match_winner = 1
        match_loser = 0
    elif p1.num_dice <= 0:
        match_winner = 0
        match_loser = 1
    else:
        match_winner = None
        match_loser = None
    
    print(f"\n{'='*50}")
    print(f"MATCH ENDED!")
    print(f"{'='*50}")
    if match_winner == human_id:
        print("🎉 YOU WON THE MATCH! 🎉")
    else:
        print("Agent won the match.")
    print(f"Total rounds played: {round_number}")
    print(f"Final score - You: {p0.num_dice} dice | Agent: {p1.num_dice} dice")
    print(f"\n[All {round_number} games saved to {data_dir}/game_trajectory.csv and game_summary.csv]")


if __name__ == "__main__":
    # Simple launcher: allow user to view rules and pick an agent
    # create config at runtime with rng_seed=None so dice are different each run
    cfg = GameConfig(dice_distribution=(5, 5), rng_seed=None)
    print("Welcome to Liar's Dice (CLI)")
    while True:
        print("\nMenu:\n  1) Show rules\n  2) Play\n  3) Quit")
        sel = input("Choose: ").strip()
        if sel == "1":
            show_rules(cfg)
            continue
        if sel == "2":
            agent_choice = "random"
            if len(sys.argv) > 1:
                agent_choice = sys.argv[1]
            else:
                # allow picking agent by name
                a = input("Choose agent (random) [default random]: ").strip()
                if a:
                    agent_choice = a
            try:
                play_against(agent_choice, config=cfg)
            except KeyboardInterrupt:
                print("\nExiting play loop.")
            break
        if sel == "3":
            print("Goodbye")
            break
        print("Unknown choice")
