"""
run_tournament_full_game.py

Run a round-robin tournament where each pairing plays full matches.
A full match consists of multiple rounds; after each round the loser
loses one die. The match ends when a player has zero dice.

Usage: python scripts/run_tournament_full_game.py --agents all --games 10 --data-dir data
"""
import os
import argparse
import datetime
import itertools
import hashlib
from collections import defaultdict
from typing import List, Any, Dict, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    _PLOTTING_AVAILABLE = False

from liars_dice.persistence import csv_io
from liars_dice.agents import AGENT_MAP
from liars_dice.core.config import GameConfig
from liars_dice.core.engine import GameEngine, IllegalMoveError
from liars_dice.core.reward import get_reward


def generate_game_id(agent0_cls, agent1_cls, timestamp: str) -> str:
    raw = f"{timestamp}_{agent0_cls.__name__}_{agent1_cls.__name__}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_full_match(agent0_cls, agent1_cls, cfg: GameConfig, game_index: int, game_id: str, timestamp: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a full match (multiple rounds) between two agent classes.

    Returns (summary_row, trajectory_rows) compatible with csv_io headers.
    """
    engine = GameEngine(cfg)
    a0 = agent0_cls()
    a1 = agent1_cls()

    error = None
    total_rounds = 0
    total_steps = 0
    total_bids = 0
    total_calls = 0
    total_bluffs_called = 0
    end_reason = None

    trajectory_rows = []

    try:
        # Play rounds until one player has zero dice
        while True:
            p0, p1 = engine.state.players
            if p0.num_dice <= 0 or p1.num_dice <= 0:
                break

            engine.start_new_round()
            total_rounds += 1

            steps = 0
            max_steps = getattr(cfg, 'max_turns', 1000)
            try:
                while not engine.is_terminal() and steps < max_steps:
                    current = engine.state.public.current_player
                    view = engine.get_view(current)
                    agent = a0 if current == 0 else a1
                    action = agent.choose_action(view)
                    state_repr = str(view)
                    action_repr = str(action)
                    engine.apply_action(current, action)
                    steps += 1
                    popped = engine.pop_events()
                    for ev in popped:
                        t = ev.get('type')
                        if t == 'BidPlaced':
                            total_bids += 1
                        elif t == 'LiarCalled':
                            total_calls += 1
                        elif t == 'RoundEnded':
                            was_true = ev.get('was_true')
                            if was_true is False:
                                total_bluffs_called += 1
                        t_str = t if t is not None else 'Unknown'
                        r = get_reward(t_str, view, action, current, engine.state.public)
                        trajectory_rows.append({
                            'game_id': game_id,
                            'round': engine.state.public.round_index,
                            'event_type': t_str,
                            'turn_index': engine.state.public.turn_index,
                            'player': ev.get('player', current),
                            'player_type': agent.__class__.__name__,
                            'payload': str(ev.get('bid', ev)),
                            'timestamp': timestamp,
                            'state': state_repr,
                            'action': action_repr,
                            'reward': r,
                        })
            except IllegalMoveError as e:
                error = str(e)
                end_reason = 'IllegalMoveError'
                current = engine.state.public.current_player
                r = get_reward('Error', engine.get_view(current), 'Error', current, engine.state.public)
                trajectory_rows.append({
                    'game_id': game_id,
                    'round': engine.state.public.round_index,
                    'event_type': 'Error',
                    'turn_index': engine.state.public.turn_index,
                    'player': current,
                    'player_type': (a0 if current == 0 else a1).__class__.__name__,
                    'payload': str(e),
                    'timestamp': timestamp,
                    'state': str(engine.get_view(current)),
                    'action': 'Error',
                    'reward': r,
                })
                engine.state.public.status = 'ENDED'
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error = f'UnexpectedException: {e}'
                end_reason = 'UnexpectedException'
                current = engine.state.public.current_player
                r = get_reward('Error', engine.get_view(current), 'Error', current, engine.state.public)
                trajectory_rows.append({
                    'game_id': game_id,
                    'round': engine.state.public.round_index,
                    'event_type': 'Error',
                    'turn_index': engine.state.public.turn_index,
                    'player': current,
                    'player_type': (a0 if current == 0 else a1).__class__.__name__,
                    'payload': tb,
                    'timestamp': timestamp,
                    'state': str(engine.get_view(current)),
                    'action': 'Error',
                    'reward': r,
                })
                engine.state.public.status = 'ENDED'

            final = engine.pop_events()
            for ev in final:
                t = ev.get('type')
                player_for_view = ev.get('player', 0)
                t_str = t if t is not None else 'Unknown'
                r = get_reward(t_str, engine.get_view(player_for_view), 'FinalEvent', player_for_view, engine.state.public)
                trajectory_rows.append({
                    'game_id': game_id,
                    'round': engine.state.public.round_index,
                    'event_type': t_str,
                    'turn_index': engine.state.public.turn_index,
                    'player': ev.get('player', None),
                    'player_type': None,
                    'payload': str(ev.get('bid', ev)),
                    'timestamp': timestamp,
                    'state': str(engine.get_view(player_for_view)),
                    'action': 'FinalEvent',
                    'reward': r,
                })

            total_steps += steps

            if engine.state.public.winner is not None:
                loser = engine.state.public.loser
                engine.state.players[loser].num_dice = max(0, engine.state.players[loser].num_dice - 1)
                trajectory_rows.append({
                    'game_id': game_id,
                    'round': engine.state.public.round_index,
                    'event_type': 'DiceLost',
                    'turn_index': engine.state.public.turn_index,
                    'player': loser,
                    'player_type': None,
                    'payload': f'player {loser} lost a die, now has {engine.state.players[loser].num_dice}',
                    'timestamp': timestamp,
                    'state': str(engine.get_view(loser)),
                    'action': 'DiceLost',
                    'reward': 0,
                })

            if total_rounds > 1000:
                end_reason = end_reason or 'match_round_limit_reached'
                break

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
            if end_reason is None:
                end_reason = 'unknown'

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error = f'UnexpectedExceptionDuringMatch: {e}'
        end_reason = 'UnexpectedExceptionDuringMatch'
        trajectory_rows.append({
            'game_id': game_id,
            'round': engine.state.public.round_index,
            'event_type': 'Error',
            'turn_index': engine.state.public.turn_index,
            'player': None,
            'player_type': None,
            'payload': tb,
            'timestamp': timestamp,
            'state': str(engine.get_view(0)),
            'action': 'Error',
            'reward': 0,
        })
        match_winner = None
        match_loser = None

    summary_row = {
        'game_id': game_id,
        'game_index': game_index,
        'timestamp': timestamp,
        'agent0': agent0_cls.__name__,
        'agent1': agent1_cls.__name__,
        'winner': match_winner,
        'loser': match_loser,
        'steps': total_steps,
        'bids': total_bids,
        'calls': total_calls,
        'bluffs_called': total_bluffs_called,
        'error': error,
        'end_reason': end_reason,
        'starting_dice_per_player': cfg.total_dice,
        'rounds_played': total_rounds,
    }

    return summary_row, trajectory_rows


def aggregate_and_plot(agent_stats: Dict[str, dict], out_path: str):
    agents = sorted(agent_stats.keys())
    wins = [agent_stats[a].get('wins', 0) for a in agents]
    games = [agent_stats[a].get('games', 0) for a in agents]
    win_perc = [(w / g * 100.0) if g > 0 else 0.0 for w, g in zip(wins, games)]

    if not _PLOTTING_AVAILABLE:
        print(f"matplotlib not available; skipping plot generation: {out_path}")
        return

    width = max(6, int(len(agents) * 0.6))
    plt.figure(figsize=(width, 4))
    bars = plt.bar(agents, win_perc, color='C0')
    plt.ylabel('Win percentage (%)')
    plt.ylim(0, 100)
    plt.title('Tournament: win% per agent')
    for rect, val in zip(bars, win_perc):
        plt.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 1.0, f"{val:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_agent_list(s: str) -> List[str]:
    if s.strip().lower() == 'all':
        return sorted(list(AGENT_MAP.keys()))
    return [x.strip() for x in s.split(',') if x.strip()]


def run_tournament(agent_keys: List[str], games_per_pair: int, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    summary_csv = os.path.join(data_dir, 'game_summary.csv')
    trajectory_csv = os.path.join(data_dir, 'game_trajectory.csv')
    tournament_csv = os.path.join(data_dir, 'tournament_summary.csv')
    agent_csv = os.path.join(data_dir, 'agent_stats.csv')
    chart_png = os.path.join(data_dir, 'win_percentages.png')

    cfg = GameConfig()

    agent_stats = defaultdict(lambda: defaultdict(int))
    tournament_rows = []

    summary_header = csv_io.get_summary_header()
    trajectory_header = csv_io.get_trajectory_header()

    pairs = list(itertools.product(agent_keys, agent_keys))
    total_games = len(pairs) * games_per_pair
    game_counter = 0
    timestamp_base = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for (a0_key, a1_key) in pairs:
        a0_cls = AGENT_MAP[a0_key]
        a1_cls = AGENT_MAP[a1_key]
        pair_wins = {'a0': 0, 'a1': 0}
        pair_steps = 0
        pair_bids = 0
        pair_calls = 0
        pair_bluffs = 0
        pair_errors = 0
        pair_beginner_wins = 0

        for i in range(games_per_pair):
            game_counter += 1
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            game_id = generate_game_id(a0_cls, a1_cls, f"{ts}_{i}")
            print(f"Running {game_counter}/{total_games}: {a0_key} (0) vs {a1_key} (1) match {i+1}/{games_per_pair}...", end=' ')
            summary_row, trajectory_rows = run_full_match(a0_cls, a1_cls, cfg, i, game_id, ts)
            # persist
            csv_io.append_row_to_csv(summary_row, summary_csv, summary_header)
            if trajectory_rows:
                csv_io.append_rows_to_csv(trajectory_rows, trajectory_csv, trajectory_header)

            winner = summary_row.get('winner')
            if winner is None:
                pair_errors += 1
            else:
                if winner == 0:
                    pair_wins['a0'] += 1
                    pair_beginner_wins += 1
                    agent_stats[a0_key]['wins'] += 1
                    agent_stats[a0_key]['wins_as_start'] += 1
                elif winner == 1:
                    pair_wins['a1'] += 1
                    agent_stats[a1_key]['wins'] += 1
                    agent_stats[a1_key]['wins_as_second'] += 1
                agent_stats[a0_key]['games'] += 1
                agent_stats[a1_key]['games'] += 1

            pair_steps += int(summary_row.get('steps') or 0)
            pair_bids += int(summary_row.get('bids') or 0)
            pair_calls += int(summary_row.get('calls') or 0)
            pair_bluffs += int(summary_row.get('bluffs_called') or 0)
            if summary_row.get('error'):
                pair_errors += 1

            print('done')

        games_played = games_per_pair
        row = {
            'timestamp': timestamp_base,
            'agent0': a0_key,
            'agent1': a1_key,
            'games': games_played,
            'wins_agent0': pair_wins['a0'],
            'wins_agent1': pair_wins['a1'],
            'beginner_win_ratio': (pair_beginner_wins / games_played) if games_played > 0 else 0.0,
            'avg_steps': (pair_steps / games_played) if games_played > 0 else 0.0,
            'total_bids': pair_bids,
            'total_calls': pair_calls,
            'total_bluffs_called': pair_bluffs,
            'errors': pair_errors,
        }
        tournament_rows.append(row)

    # write tournament summary
    tour_header = ['timestamp', 'agent0', 'agent1', 'games', 'wins_agent0', 'wins_agent1', 'beginner_win_ratio', 'avg_steps', 'total_bids', 'total_calls', 'total_bluffs_called', 'errors']
    csv_io_append = csv_io
    # use simple writer for tournament and agent CSVs
    def write_rows_to_csv(rows: List[dict], path: str, header: List[str]):
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write_rows_to_csv(tournament_rows, tournament_csv, tour_header)

    # agent aggregates
    agent_rows = []
    agent_header = ['agent', 'games', 'wins', 'win_percent', 'wins_as_start', 'wins_as_second']
    for agent in sorted(agent_keys):
        g = agent_stats[agent].get('games', 0)
        w = agent_stats[agent].get('wins', 0)
        win_percent = (w / g * 100.0) if g > 0 else 0.0
        agent_rows.append({
            'agent': agent,
            'games': g,
            'wins': w,
            'win_percent': f"{win_percent:.3f}",
            'wins_as_start': agent_stats[agent].get('wins_as_start', 0),
            'wins_as_second': agent_stats[agent].get('wins_as_second', 0),
        })
    write_rows_to_csv(agent_rows, agent_csv, agent_header)

    # plot
    aggregate_and_plot(agent_stats, chart_png)

    print(f"Tournament finished. Game summaries saved to {summary_csv}, trajectories to {trajectory_csv}")
    print(f"Tournament summary: {tournament_csv}")
    print(f"Per-agent stats: {agent_csv}")
    print(f"Win percentage chart: {chart_png}")


def main():
    parser = argparse.ArgumentParser(description='Run a round-robin 1v1 tournament of full matches between agents')
    parser.add_argument('--agents', type=str, default='all', help='Comma-separated list of agent keys from AGENT_MAP or "all"')
    parser.add_argument('--games', type=int, default=10, help='Number of matches per ordered pairing')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save csv and charts')
    args = parser.parse_args()

    agent_keys = parse_agent_list(args.agents)
    unknown = [a for a in agent_keys if a not in AGENT_MAP]
    if unknown:
        raise SystemExit(f"Unknown agents: {unknown}. Supported: {list(AGENT_MAP.keys())}")

    run_tournament(agent_keys, args.games, args.data_dir)


if __name__ == '__main__':
    main()

