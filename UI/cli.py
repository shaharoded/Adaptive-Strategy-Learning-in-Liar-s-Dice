import sys
from typing import Optional

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.actions import BidAction, CallLiarAction, Action
from liars_dice.core.bid import Bid
from liars_dice.agents.random_agent import RandomAgent
from liars_dice.agents.base import Agent



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
    if name in ("random", "r"):
        return RandomAgent()
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
    Play a single round of Liar's Dice as a human (player 0) against an agent (player 1) in the CLI.
    Args:
        agent_name (str): Name of the agent to play against (default 'random').
        config (GameConfig, optional): Game configuration. If None, uses default config.
    """
    # Setup game: create config at runtime so rng_seed can be None (non-deterministic)
    if config is None:
        config = GameConfig(dice_distribution=(5, 5), rng_seed=None)
    engine = GameEngine(config)
    agent = choose_agent(agent_name)

    engine.start_new_round()
    # Let human be player 0 and agent be player 1
    human_id = 0
    agent_id = 1

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
            except IllegalMoveError as e:
                # If agent made illegal move, treat as pass/call liar
                print(f"Agent made illegal move: {e}. Agent will call liar instead.")
                engine.apply_action(agent_id, CallLiarAction())

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
