import tkinter as tk
from tkinter import messagebox
from typing import Optional
import os
import datetime
import hashlib
from PIL import Image, ImageTk  # For loading banner image

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid
from liars_dice.core.reward import get_reward
from liars_dice.agents import AGENT_MAP
from liars_dice.persistence import csv_io


class DiceCanvas(tk.Canvas):
    """A small canvas widget that draws a dice face (pips) or a back-face when face is None."""

    # pip positions for a normalized 0..1 square
    _POSITIONS = {
        1: [(0.5, 0.5)],
        2: [(0.25, 0.25), (0.75, 0.75)],
        3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
        4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
        5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
        6: [(0.25, 0.2), (0.25, 0.5), (0.25, 0.8), (0.75, 0.2), (0.75, 0.5), (0.75, 0.8)],
    }

    def __init__(self, master, size: int = 56, face: Optional[int] = None, **kwargs):
        super().__init__(master, width=size, height=size, bg=master.cget("bg"), highlightthickness=0, **kwargs)
        self.size = size
        # allow None for hidden/backside
        self.face = face
        self._draw_face(self.face)

    def _draw_face(self, face: Optional[int]):
        self.delete("all")
        s = self.size
        pad = max(4, s // 10)
        # draw background rectangle
        if face is None:
            # back-face appearance
            self.create_rectangle(pad, pad, s - pad, s - pad, fill="#666", outline="#333", width=2)
            # draw a subtle pattern (three dots) or question mark
            try:
                self.create_text(s/2, s/2, text="?", fill="white", font=("Helvetica", max(12, s//2), "bold"))
            except Exception:
                # fallback if font size invalid
                self.create_text(s/2, s/2, text="?", fill="white")
            return

        # draw rounded-ish rect for face-up dice
        self.create_rectangle(pad, pad, s - pad, s - pad, fill="white", outline="#333", width=2)
        # draw pips
        positions = self._POSITIONS.get(face, [])
        pip_r = max(3, s // 12)
        for (nx, ny) in positions:
            x = pad + nx * (s - 2 * pad)
            y = pad + ny * (s - 2 * pad)
            self.create_oval(x - pip_r, y - pip_r, x + pip_r, y + pip_r, fill="#111", outline="")

    def set_face(self, face: Optional[int]):
        self.face = face
        self._draw_face(face)


class LiarDiceGUI:
    def __init__(self, root: tk.Tk, config: Optional[GameConfig] = None):
        self.root = root
        root.title("Liar's Dice - Friendly GUI")

        self.config = config or GameConfig(dice_distribution=(5, 5), rng_seed=None)
        self.engine: Optional[GameEngine] = None
        self.agent = None
        # The human is Player 1 (so we refer to them as "You"). The agent is Player 0.
        self.human_id = 1
        self.agent_id = 0
        
        # Match state tracking
        self.match_active = False  # Track if a match is currently in progress
        self.total_rounds = 0
        # Per-round tracking (reset each round)
        self.round_steps = 0
        self.round_bids = 0
        self.round_calls = 0

        # Track displayed history entries (bids and calls) so we don't overwrite them
        self._displayed_entries = []  # list of str
        self._bid_actors = []  # parallel list for which player made each bid (only for bids)

        # CSV tracking variables
        self.game_id = None
        self.timestamp = None
        self.trajectory_rows = []
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Header section with side banners
        header_container = tk.Frame(root)
        header_container.pack(fill=tk.X, padx=0, pady=0)
        
        # Load and display banners with fixed dimensions
        banner_width = 130
        banner_height = 160
        
        # Left banner
        try:
            banner_left_path = os.path.join("UI", "images", "banner_left.png")
            if os.path.exists(banner_left_path):
                img_left = Image.open(banner_left_path)
                img_left = img_left.resize((banner_width, banner_height), Image.Resampling.LANCZOS)
                self.banner_left_photo = ImageTk.PhotoImage(img_left)
                banner_left = tk.Label(header_container, image=self.banner_left_photo)
                banner_left.pack(side=tk.LEFT, padx=(10, 0))
        except Exception as e:
            print(f"Could not load left banner: {e}")
        
        # Top controls: agent selection and start
        top = tk.Frame(header_container)
        top.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=8)
        
        # Title row
        title_row = tk.Frame(top)
        title_row.pack(fill=tk.X, pady=(0, 50))
        tk.Label(title_row, text="🎲 Liar's Dice 🎲", font=("Helvetica", 22, "bold")).pack()
        
        # Controls row
        controls_row = tk.Frame(top)
        controls_row.pack(fill=tk.X)
        tk.Label(controls_row, text="Opponent:", font=("Helvetica", 10)).pack(side=tk.LEFT)
        # Get all available agents from AGENT_MAP
        agent_names = sorted([name.title() for name in AGENT_MAP.keys()])
        default_agent = agent_names[0] if agent_names else "Random"
        self.agent_var = tk.StringVar(value=default_agent)
        self.agent_menu = tk.OptionMenu(controls_row, self.agent_var, *agent_names)
        self.agent_menu.pack(side=tk.LEFT, padx=(6, 12))
        self.start_button = tk.Button(controls_row, text="Start Match", command=self.start_match)
        self.start_button.pack(side=tk.LEFT)
        # Stop Match button (initially disabled)
        self.stop_button = tk.Button(controls_row, text="Stop Match", command=self.stop_match, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        
        # Right banner
        try:
            banner_right_path = os.path.join("UI", "images", "banner_right.png")
            if os.path.exists(banner_right_path):
                img_right = Image.open(banner_right_path)
                img_right = img_right.resize((banner_width, banner_height), Image.Resampling.LANCZOS)
                self.banner_right_photo = ImageTk.PhotoImage(img_right)
                banner_right = tk.Label(header_container, image=self.banner_right_photo)
                banner_right.pack(side=tk.LEFT, padx=(0, 10))
        except Exception as e:
            print(f"Could not load right banner: {e}")

        # Match info (rounds, dice counts)
        match_info_frame = tk.Frame(root)
        match_info_frame.pack(padx=8, pady=4, fill=tk.X)
        
        self.match_label = tk.Label(match_info_frame, text="Match: Not started", font=("Helvetica", 10, "bold"))
        self.match_label.pack(side=tk.LEFT, padx=(0, 12))
        
        self.dice_status_label = tk.Label(match_info_frame, text="Your dice: - | Agent dice: -", font=("Helvetica", 10))
        self.dice_status_label.pack(side=tk.LEFT, padx=(0, 12))

        # Info & status
        info_frame = tk.Frame(root)
        info_frame.pack(padx=8, pady=4, fill=tk.X)

        self.round_label = tk.Label(info_frame, text="Round: -", font=("Helvetica", 10, "bold"))
        self.round_label.pack(side=tk.LEFT, padx=(0, 12))

        self.turn_label = tk.Label(info_frame, text="Turn: -", font=("Helvetica", 10))
        self.turn_label.pack(side=tk.LEFT, padx=(0, 12))

        self.current_label = tk.Label(info_frame, text="Current: -", font=("Helvetica", 10))
        self.current_label.pack(side=tk.LEFT)

        # Main area: your dice + actions, opponent & bid history
        main = tk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Left: your dice and bid inputs
        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        dice_frame = tk.LabelFrame(left, text="Your dice")
        dice_frame.pack(fill=tk.X, padx=6, pady=6)
        self.dice_container = tk.Frame(dice_frame)
        self.dice_container.pack(padx=6, pady=6)

        action_frame = tk.LabelFrame(left, text="Actions")
        action_frame.pack(fill=tk.X, padx=6, pady=6)

        bid_inputs = tk.Frame(action_frame)
        bid_inputs.pack(side=tk.LEFT, padx=6)
        tk.Label(bid_inputs, text="Quantity:").grid(row=0, column=0)
        self.qty_entry = tk.Entry(bid_inputs, width=6)
        self.qty_entry.grid(row=0, column=1, padx=(4, 12))
        tk.Label(bid_inputs, text="Face:").grid(row=0, column=2)
        self.face_entry = tk.Entry(bid_inputs, width=6)
        self.face_entry.grid(row=0, column=3, padx=(4, 12))

        self.bid_button = tk.Button(action_frame, text="Bid", command=self.on_bid, state=tk.DISABLED)
        self.bid_button.pack(side=tk.LEFT, padx=(6, 12))

        self.call_button = tk.Button(action_frame, text="Call Liar", command=self.on_call, state=tk.DISABLED)
        self.call_button.pack(side=tk.LEFT, padx=(6, 12))

        # Right: opponent info and bid history
        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))

        opp_frame = tk.LabelFrame(right, text="Opponent")
        opp_frame.pack(fill=tk.X, padx=6, pady=6)
        # Opponent is player 0 (agent)
        self.opp_label = tk.Label(opp_frame, text=f"Player {self.agent_id} (Agent)")
        self.opp_label.pack(anchor="w", padx=6, pady=2)
        # replaced textual dice count with a small container that shows a row of hidden dice tiles
        self.opp_dice_container = tk.Frame(opp_frame)
        self.opp_dice_container.pack(anchor="w", padx=6, pady=2)

        bidhist_frame = tk.LabelFrame(right, text="Bid history")
        bidhist_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.bid_listbox = tk.Listbox(bidhist_frame, height=8)
        self.bid_listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Status message
        self.status_var = tk.StringVar(value="Choose an opponent and Start Match")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=(0, 8))

        # engine will be initialized when Start Match is pressed

    def start_match(self):
        """Start a full match (multiple rounds until one player has zero dice)"""
        # Lock agent selection during match
        choice = (self.agent_var.get() or "").lower()
        if choice in AGENT_MAP:
            self.agent = AGENT_MAP[choice]()
        else:
            # fallback to first available agent
            first_agent = next(iter(AGENT_MAP.values()))
            self.agent = first_agent()
        
        # Initialize engine and match tracking
        self.engine = GameEngine(self.config)
        self.match_active = True
        self.total_rounds = 0
        
        # Disable agent menu and start button, enable stop button
        self.agent_menu.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start the first round
        self.start_new_round()
    
    def stop_match(self):
        """Stop the current match"""
        if not self.match_active:
            return
        
        result = messagebox.askyesno("Stop Match", "Are you sure you want to stop the current match?")
        if result:
            self.match_active = False
            
            # Re-enable agent menu and start button, disable stop button
            self.agent_menu.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            self.status_var.set("Match stopped. Choose an opponent and Start Match to play again.")
            self.match_label.config(text="Match: Stopped")

    def start_new_round(self):
        """Start a new round within the current match"""
        if self.engine is None:
            return
        
        # Check if match should end (any player has zero dice)
        p0, p1 = self.engine.state.players
        if p0.num_dice <= 0 or p1.num_dice <= 0:
            self.end_match()
            return
        
        self.engine.start_new_round()
        self.total_rounds += 1
        
        # Generate unique game_id for this round
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        agent_name = self.agent.__class__.__name__ if self.agent else "Unknown"
        raw_id = f"gui_{self.timestamp}_{os.getpid()}_{agent_name}_round{self.total_rounds}"
        self.game_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        self.trajectory_rows = []  # Fresh trajectory for this round
        
        # Reset per-round counters
        self.round_steps = 0
        self.round_bids = 0
        self.round_calls = 0
        
        # clear bid history for new round
        self.bid_listbox.delete(0, tk.END)
        self._displayed_entries.clear()
        self._bid_actors.clear()
        
        # Record round start events
        self._record_event("RoundStarted", {"round": self.engine.state.public.round_index}, 
                          player_type=None, player=None, state=self.engine.get_view(0), 
                          action=None, reward_val=0)
        p0, p1 = self.engine.state.players
        self._record_event("DiceRolled", {"player0": p0.private_dice.copy(), "player1": p1.private_dice.copy()},
                          player_type=None, player=None, state=self.engine.get_view(0),
                          action=None, reward_val=0)
        
        self.match_label.config(text=f"Match: Round {self.total_rounds}")
        self.update_ui()
        # If agent starts, schedule its move
        self.root.after(200, self.maybe_agent_move)
    
    def end_match(self):
        """End the match and show final results"""
        self.match_active = False
        p0, p1 = self.engine.state.players
        
        # Determine match winner
        if p1.num_dice <= 0:  # Human lost
            result_msg = f"Match Over!\n\nAgent wins the match!\n\nRounds played: {self.total_rounds}"
        elif p0.num_dice <= 0:  # Agent lost
            result_msg = f"Match Over!\n\n🎉 YOU WIN THE MATCH! 🎉\n\nRounds played: {self.total_rounds}"
        else:
            result_msg = f"Match ended.\n\nRounds played: {self.total_rounds}"
        
        # Re-enable controls
        self.agent_menu.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.status_var.set("Match ended. Start a new match to play again.")
        self.match_label.config(text="Match: Ended")
        
        messagebox.showinfo("Match Result", result_msg + f"\n\nAll {self.total_rounds} games saved to {self.data_dir}/")
    
    def _save_match_data(self, end_reason, match_winner=None, match_loser=None):
        """Save match trajectory and summary to CSV files"""
        # Write trajectory to CSV
        trajectory_csv = os.path.join(self.data_dir, "match_trajectory.csv")
        trajectory_header = csv_io.get_trajectory_header()
        csv_io.append_rows_to_csv(self.trajectory_rows, trajectory_csv, trajectory_header)
        
        # Write summary to CSV
        summary_csv = os.path.join(self.data_dir, "match_summary.csv")
        summary_header = csv_io.get_summary_header()
        
        agent_name = self.agent.__class__.__name__ if self.agent else "Unknown"
        summary_row = {
            "game_id": self.game_id,
            "game_index": None,
            "timestamp": self.timestamp,
            "agent0": agent_name,  # agent is player 0
            "agent1": "Human",     # human is player 1
            "winner": match_winner,
            "loser": match_loser,
            "starting_dice_per_player": 5,
            "rounds_played": self.total_rounds,
            "steps": self.total_steps,
            "bids": self.total_bids,
            "calls": self.total_calls,
            "bluffs_called": 0,  # Can be enhanced
            "error": None,
            "end_reason": end_reason,
        }
        csv_io.append_row_to_csv(summary_row, summary_csv, summary_header)

    def _record_event(self, event_type, payload, player_type=None, player=None, state=None, action=None, reward_val=None):
        """Record a trajectory event for CSV writing."""
        if self.engine is None:
            return
        r = reward_val if reward_val is not None else get_reward(
            event_type, state, action, player, self.engine.state.public
        )
        self.trajectory_rows.append({
            "game_id": self.game_id,
            "round": self.engine.state.public.round_index,
            "event_type": event_type,
            "turn_index": self.engine.state.public.turn_index,
            "player": player,
            "player_type": player_type,
            "payload": str(payload),
            "timestamp": self.timestamp,
            "state": str(state) if state is not None else "",
            "action": str(action) if action is not None else "",
            "reward": r,
        })

    def update_ui(self):
        if self.engine is None:
            return
        public = self.engine.state.public
        p = self.engine.state.players[self.human_id]
        p_agent = self.engine.state.players[self.agent_id]
        
        self.round_label.config(text=f"Round: {self.total_rounds}")
        self.turn_label.config(text=f"Turn: {public.turn_index}")
        self.current_label.config(text=f"Current: Player {public.current_player}")
        
        # Update dice status
        self.dice_status_label.config(text=f"Your dice: {p.num_dice} | Agent dice: {p_agent.num_dice}")

        # draw dice graphically for human
        for child in self.dice_container.winfo_children():
            child.destroy()
        for d in p.private_dice:
            dc = DiceCanvas(self.dice_container, size=56, face=d)
            dc.pack(side=tk.LEFT, padx=4)

        # opponent dice: render hidden/back-face tiles for each opponent die
        opp = self.engine.state.players[self.agent_id]
        for child in self.opp_dice_container.winfo_children():
            child.destroy()
        for _ in range(opp.num_dice):
            dc = DiceCanvas(self.opp_dice_container, size=40, face=None)
            dc.pack(side=tk.LEFT, padx=3)

        # show last bid
        last = public.last_bid
        if last is None:
            # don't clear the custom history here; just update the status guidance
            self.status_var.set("No bids yet. Make an opening bid or wait for opponent.")
        else:
            self.status_var.set(f"Current bid: {last.quantity} x {last.face}")

        # disable/enable controls depending on whose turn and match status
        if public.current_player != self.human_id or public.status != "BIDDING" or not self.match_active:
            self.bid_button.config(state=tk.DISABLED)
            self.call_button.config(state=tk.DISABLED)
            self.qty_entry.config(state=tk.DISABLED)
            self.face_entry.config(state=tk.DISABLED)
        else:
            self.bid_button.config(state=tk.NORMAL)
            self.call_button.config(state=tk.NORMAL)
            self.qty_entry.config(state=tk.NORMAL)
            self.face_entry.config(state=tk.NORMAL)

        # If the round ended, show outcome
        if public.status == "ENDED":
            self.on_round_ended()

    def on_bid(self):
        if self.engine is None:
            return
        # read entries
        q_s = self.qty_entry.get().strip()
        f_s = self.face_entry.get().strip()
        try:
            qty = int(q_s)
            face = int(f_s)
        except Exception:
            self.status_var.set("Please enter valid integers for quantity and face.")
            return
        bid = Bid(qty, face)
        try:
            bid.validate(self.config)
        except Exception as e:
            self.status_var.set(f"Invalid bid: {e}")
            return
        # check higher than last
        last = self.engine.state.public.last_bid
        if not bid.is_higher_than(last):
            self.status_var.set("Bid must be higher than last bid.")
            return
        try:
            action = BidAction(bid)
            state = self.engine.get_view(self.human_id)
            self.engine.apply_action(self.human_id, action)
            self.round_steps += 1
            self.round_bids += 1
            # Human is Player 1 -> use 'You' phrasing
            entry = f"You bid: {qty} x {face}"
            self.bid_listbox.insert(tk.END, entry)
            self._displayed_entries.append(entry)
            self._bid_actors.append(self.human_id)
            self.status_var.set(entry)
            
            # Record bid event
            self._record_event("BidPlaced", {"bid": str(bid)},
                             player_type="Human", player=self.human_id,
                             state=state, action=action)
        except IllegalMoveError as e:
            self.status_var.set(f"Illegal move: {e}")
            self._record_event("Error", str(e),
                             player_type="Human", player=self.human_id,
                             state=self.engine.get_view(self.human_id), action="Error")
        self.update_ui()
        # schedule agent move
        self.root.after(300, self.maybe_agent_move)

    def on_call(self):
        if self.engine is None:
            return
        try:
            action = CallLiarAction()
            state = self.engine.get_view(self.human_id)
            self.engine.apply_action(self.human_id, action)
            self.round_steps += 1
            self.round_calls += 1
            entry = "You called liar"
            self.bid_listbox.insert(tk.END, entry)
            self._displayed_entries.append(entry)
            self.status_var.set(entry)
            
            # Record call liar event
            self._record_event("LiarCalled", {"caller": self.human_id},
                             player_type="Human", player=self.human_id,
                             state=state, action=action)
        except IllegalMoveError as e:
            self.status_var.set(f"Illegal move: {e}")
            self._record_event("Error", str(e),
                             player_type="Human", player=self.human_id,
                             state=self.engine.get_view(self.human_id), action="Error")
        self.update_ui()
        self.root.after(300, self.maybe_agent_move)

    def maybe_agent_move(self):
        if self.engine is None:
            return
        public = self.engine.state.public
        if public.status != "BIDDING":
            return
        if public.current_player == self.agent_id:
            # disable human controls while agent thinks
            self.bid_button.config(state=tk.DISABLED)
            self.call_button.config(state=tk.DISABLED)
            self.qty_entry.config(state=tk.DISABLED)
            self.face_entry.config(state=tk.DISABLED)
            # run agent move after a short delay to show it
            self.root.after(400, self.agent_move)

    def agent_move(self):
        if self.engine is None:
            return
        if self.engine.state.public.current_player != self.agent_id:
            return
        view = self.engine.get_view(self.agent_id)
        action = self.agent.choose_action(view)
        agent_name = self.agent.__class__.__name__
        try:
            self.engine.apply_action(self.agent_id, action)
            self.round_steps += 1
            # create friendly messages for agent actions
            if isinstance(action, BidAction):
                self.round_bids += 1
                b = action.bid
                # Agent is player 0 -> reference by player number
                entry = f"Player {self.agent_id} bid: {b.quantity} x {b.face}"
                self.bid_listbox.insert(tk.END, entry)
                self._displayed_entries.append(entry)
                self._bid_actors.append(self.agent_id)
                self.status_var.set(entry)
                
                # Record bid event
                self._record_event("BidPlaced", {"bid": str(b)},
                                 player_type=agent_name, player=self.agent_id,
                                 state=view, action=action)
            else:
                self.round_calls += 1
                entry = f"Player {self.agent_id} called liar"
                self.bid_listbox.insert(tk.END, entry)
                self._displayed_entries.append(entry)
                self.status_var.set(entry)
                
                # Record call liar event
                self._record_event("LiarCalled", {"caller": self.agent_id},
                                 player_type=agent_name, player=self.agent_id,
                                 state=view, action=action)
        except IllegalMoveError as e:
            # fallback: agent calls liar
            self.status_var.set(f"Agent made illegal move: {e}. Calling liar instead.")
            self._record_event("Error", str(e),
                             player_type=agent_name, player=self.agent_id,
                             state=view, action="Error")
            try:
                fallback_action = CallLiarAction()
                self.engine.apply_action(self.agent_id, fallback_action)
                self.round_steps += 1
                self.round_calls += 1
                entry = f"Player {self.agent_id} called liar"
                self.bid_listbox.insert(tk.END, entry)
                self._displayed_entries.append(entry)
                self.status_var.set(entry)
                
                # Record fallback call liar
                self._record_event("LiarCalled", {"caller": self.agent_id, "fallback": True},
                                 player_type=agent_name, player=self.agent_id,
                                 state=self.engine.get_view(self.agent_id), action=fallback_action)
            except IllegalMoveError:
                pass
        self.update_ui()
        # If next is agent again (unlikely), schedule again
        self.root.after(200, self.maybe_agent_move)

    def on_round_ended(self):
        """Handle end of a single round within the match"""
        if not self.match_active:
            return
            
        public = self.engine.state.public
        p0, p1 = self.engine.state.players
        
        # Show round result
        msg = f"Round {self.total_rounds} ended!\n\n"
        msg += f"Winner: Player {public.winner}\n"
        msg += f"Loser: Player {public.loser}\n"
        msg += f"Final bid: {public.last_bid.quantity if public.last_bid else 'N/A'} x {public.last_bid.face if public.last_bid else 'N/A'}\n\n"
        msg += f"Player 0 (Agent) dice: {tuple(p0.private_dice)}\n"
        msg += f"Player 1 (You) dice: {tuple(p1.private_dice)}\n"
        
        # Record dice reveal and round end events
        self._record_event("DiceRevealed", 
                          {"all_dice": {0: p0.private_dice, 1: p1.private_dice}},
                          player_type=None, player=None,
                          state=self.engine.get_view(0), action=None, reward_val=0)
        
        # Record RoundEnded for each player with their respective rewards
        agent_state = self.engine.get_view(self.agent_id)
        agent_reward = get_reward("RoundEnded", agent_state, None, self.agent_id, public)
        agent_name = self.agent.__class__.__name__ if self.agent else "Unknown"
        self._record_event("RoundEnded", 
                          {"winner": public.winner, "loser": public.loser,
                           "match_count": None, "was_true": None},
                          player_type=agent_name, player=self.agent_id,
                          state=agent_state, action=None, reward_val=agent_reward)
        
        human_state = self.engine.get_view(self.human_id)
        human_reward = get_reward("RoundEnded", human_state, None, self.human_id, public)
        self._record_event("RoundEnded", 
                          {"winner": public.winner, "loser": public.loser,
                           "match_count": None, "was_true": None},
                          player_type="Human", player=self.human_id,
                          state=human_state, action=None, reward_val=human_reward)
        
        # Save this round as a separate game
        trajectory_csv = os.path.join(self.data_dir, "game_trajectory.csv")
        trajectory_header = csv_io.get_trajectory_header()
        csv_io.append_rows_to_csv(self.trajectory_rows, trajectory_csv, trajectory_header)
        
        summary_csv = os.path.join(self.data_dir, "game_summary.csv")
        summary_header = csv_io.get_summary_header()
        summary_row = {
            "game_id": self.game_id,
            "game_index": self.total_rounds - 1,
            "timestamp": self.timestamp,
            "agent0": agent_name,
            "agent1": "Human",
            "winner": public.winner,
            "loser": public.loser,
            "steps": self.round_steps,
            "bids": self.round_bids,
            "calls": self.round_calls,
            "bluffs_called": 0,
            "error": None,
            "end_reason": "winner declared" if public.winner is not None else None,
        }
        csv_io.append_row_to_csv(summary_row, summary_csv, summary_header)
        
        # Decrement loser's dice
        if public.loser is not None:
            loser = public.loser
            self.engine.state.players[loser].num_dice = max(0, self.engine.state.players[loser].num_dice - 1)
            msg += f"\nPlayer {loser} loses a die! Now has {self.engine.state.players[loser].num_dice} dice."
        
        messagebox.showinfo("Round Result", msg)
        
        # Check if match continues or ends
        p0, p1 = self.engine.state.players
        if p0.num_dice <= 0 or p1.num_dice <= 0:
            # Match over
            self.end_match()
        else:
            # Continue to next round
            self.root.after(500, self.start_new_round)


def main():
    root = tk.Tk()
    app = LiarDiceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
