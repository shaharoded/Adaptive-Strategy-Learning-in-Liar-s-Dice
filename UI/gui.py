import tkinter as tk
from tkinter import messagebox
from typing import Optional

from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.actions import BidAction, CallLiarAction
from liars_dice.core.bid import Bid
from liars_dice.agents.random_agent import RandomAgent


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

        # Track displayed history entries (bids and calls) so we don't overwrite them
        self._displayed_entries = []  # list of str
        self._bid_actors = []  # parallel list for which player made each bid (only for bids)

        # Top controls: agent selection and start
        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(top, text="Opponent agent:", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.agent_var = tk.StringVar(value="Random")
        agent_menu = tk.OptionMenu(top, self.agent_var, "Random")
        agent_menu.pack(side=tk.LEFT, padx=(6, 12))
        self.start_button = tk.Button(top, text="Start Game", command=self.start_game)
        self.start_button.pack(side=tk.LEFT)
        # New Round button lives next to Start Game and remains after game started
        self.restart_button = tk.Button(top, text="New Round", command=self.start_new_round, state=tk.DISABLED)
        self.restart_button.pack(side=tk.LEFT, padx=(8, 0))

        # Info & status
        info_frame = tk.Frame(root)
        info_frame.pack(padx=8, pady=4, fill=tk.X)

        self.round_label = tk.Label(info_frame, text="Round: -", font=("Helvetica", 10, "bold"))
        self.round_label.pack(side=tk.LEFT, padx=(0, 12))

        self.turn_label = tk.Label(info_frame, text="Turn: -", font=("Helvetica", 10))
        self.turn_label.pack(side=tk.LEFT, padx=(0, 12))

        self.current_label = tk.Label(info_frame, text="Current: -", font=("Helvetica", 10))
        self.current_label.pack(side=tk.LEFT)

        # Main area: left=your dice + actions, right=opponent & bid history
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
        self.status_var = tk.StringVar(value="Choose an opponent and Start Game")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=(0, 8))

        # engine will be initialized when Start Game is pressed

    def start_game(self):
        # create the chosen agent and engine, then start round
        choice = (self.agent_var.get() or "Random").lower()
        if choice == "random":
            self.agent = RandomAgent()
        else:
            # fallback
            self.agent = RandomAgent()
        # enable restart button and hide start game
        self.engine = GameEngine(self.config)
        self.restart_button.config(state=tk.NORMAL)
        # hide the start button once game begins, leave Restart visible
        try:
            # hide the Start Game control once started
            self.start_button.pack_forget()
        except Exception:
            self.start_button.config(state=tk.DISABLED)
        self.start_new_round()

    def start_new_round(self):
        if self.engine is None:
            self.engine = GameEngine(self.config)
        self.engine.start_new_round()
        # clear bid history and actor tracking
        self.bid_listbox.delete(0, tk.END)
        self._displayed_entries.clear()
        self._bid_actors.clear()
        self.update_ui()
        # If agent starts, schedule its move
        self.root.after(200, self.maybe_agent_move)

    def update_ui(self):
        if self.engine is None:
            return
        public = self.engine.state.public
        p = self.engine.state.players[self.human_id]
        self.round_label.config(text=f"Round: {public.round_index}")
        self.turn_label.config(text=f"Turn: {public.turn_index}")
        self.current_label.config(text=f"Current: Player {public.current_player}")

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

        # disable/enable controls depending on whose turn
        if public.current_player != self.human_id or public.status != "BIDDING":
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
        if not bid.is_higher_than(last, self.config):
            self.status_var.set("Bid must be higher than last bid.")
            return
        try:
            self.engine.apply_action(self.human_id, BidAction(bid))
            # Human is Player 1 -> use 'You' phrasing
            entry = f"You bid: {qty} x {face}"
            self.bid_listbox.insert(tk.END, entry)
            self._displayed_entries.append(entry)
            self._bid_actors.append(self.human_id)
            self.status_var.set(entry)
        except IllegalMoveError as e:
            self.status_var.set(f"Illegal move: {e}")
        self.update_ui()
        # schedule agent move
        self.root.after(300, self.maybe_agent_move)

    def on_call(self):
        if self.engine is None:
            return
        try:
            self.engine.apply_action(self.human_id, CallLiarAction())
            entry = "You called liar"
            self.bid_listbox.insert(tk.END, entry)
            self._displayed_entries.append(entry)
            self.status_var.set(entry)
        except IllegalMoveError as e:
            self.status_var.set(f"Illegal move: {e}")
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
        try:
            self.engine.apply_action(self.agent_id, action)
            # create friendly messages for agent actions
            if isinstance(action, BidAction):
                b = action.bid
                # Agent is player 0 -> reference by player number
                entry = f"Player {self.agent_id} bid: {b.quantity} x {b.face}"
                self.bid_listbox.insert(tk.END, entry)
                self._displayed_entries.append(entry)
                self._bid_actors.append(self.agent_id)
                self.status_var.set(entry)
            else:
                entry = f"Player {self.agent_id} called liar"
                self.bid_listbox.insert(tk.END, entry)
                self._displayed_entries.append(entry)
                self.status_var.set(entry)
        except IllegalMoveError as e:
            # fallback: agent calls liar
            self.status_var.set(f"Agent made illegal move: {e}. Calling liar instead.")
            try:
                self.engine.apply_action(self.agent_id, CallLiarAction())
                entry = f"Player {self.agent_id} called liar"
                self.bid_listbox.insert(tk.END, entry)
                self._displayed_entries.append(entry)
                self.status_var.set(entry)
            except IllegalMoveError:
                pass
        self.update_ui()
        # If next is agent again (unlikely), schedule again
        self.root.after(200, self.maybe_agent_move)

    def on_round_ended(self):
        public = self.engine.state.public
        p0, p1 = self.engine.state.players
        msg = f"Round ended. Winner: Player {public.winner} (loser: {public.loser})\n"
        msg += f"Final bid: {public.last_bid.quantity if public.last_bid else 'N/A'} x {public.last_bid.face if public.last_bid else 'N/A'}\n"
        msg += f"Player 0 dice: {tuple(p0.private_dice)}\nPlayer 1 dice: {tuple(p1.private_dice)}"
        messagebox.showinfo("Round Result", msg)


def main():
    root = tk.Tk()
    app = LiarDiceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
