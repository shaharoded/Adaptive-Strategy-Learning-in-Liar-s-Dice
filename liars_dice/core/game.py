from .config import GameConfig
from .engine import GameEngine

class Game:
    def __init__(self, agents, config: GameConfig = None):
        if config is None:
            config = GameConfig(num_players=len(agents))

        self.config = config
        self.engine = GameEngine(config)
        self.agents = agents
        self.game_history = []

    def play_turn(self):
        if self.engine.is_terminal():
            return

        current_player_id = self.engine.state.public.current_player
        agent = self.agents[current_player_id]

        state_view = self.engine.get_view(current_player_id)

        # Store a copy of the state before the action
        self.game_history.append(self.engine.state)

        action = agent.act(state_view)
        self.engine.apply_action(current_player_id, action)

    def play_round(self):
        self.engine.start_new_round()
        self.game_history = []
        while not self.engine.is_terminal():
            self.play_turn()

    def play_match(self):
        while not self.is_game_over():
            self.play_round()
            loser = self.engine.state.public.loser
            if loser is not None:
                self.engine.state.players[loser].num_dice -= 1

    def is_game_over(self):
        return any(p.num_dice == 0 for p in self.engine.state.players)

    def get_winner(self):
        if not self.is_game_over():
            return None

        for player in self.engine.state.players:
            if player.num_dice > 0:
                return player.player_id
        return None

    def get_game_history(self):
        return self.game_history

