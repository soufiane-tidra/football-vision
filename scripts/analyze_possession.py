"""Ball possession, passes and turnovers.

Requires:
    python -m scripts.build_players   -> players.csv
    python -m scripts.track_ball      -> ball.csv

Run:
    python -m scripts.analyze_possession

Output:
    data/processed/possession.csv   who had the ball, from which frame to which frame
    data/processed/events.csv       passes and turnovers
"""

import csv
from collections import Counter
from pathlib import Path

from src.analytics.possession import (
    detect_events,
    owner_per_frame,
    passing_network,
    possession_spells,
    team_possession,
)
from src.ball.ball_io import load_ball
from src.tracking.players_io import load_players
from src.utils.config import load_config
from src.video.video import get_video_info


def main():
    config = load_config()
    fps = get_video_info(config.video)["fps"]

    players, info = load_players(config.paths.players)
    ball = load_ball(config.paths.ball)

    players_by_frame = {}
    for player_id, track in players.items():
        if info[player_id].role == "referee":
            continue
        for frame, box in zip(track.frames, track.boxes):
            players_by_frame.setdefault(frame, []).append((player_id, box))

    team_of = {player_id: details.team for player_id, details in info.items()}

    owners = owner_per_frame(players_by_frame, {f: (b["x"], b["y"]) for f, b in ball.items()})
    # A touch must last ~0.25 s to count as control; shorter ones are duels or the
    # ball rolling past a foot. Dribble gaps up to 0.5 s are bridged.
    spells = possession_spells(
        owners, team_of,
        max_bridge_frames=int(0.5 * fps),
        min_spell_frames=max(2, int(0.25 * fps)),
    )
    events = detect_events(spells, max_gap_frames=int(3 * fps))

    possession_path = Path(config.paths.possession)
    possession_path.parent.mkdir(parents=True, exist_ok=True)

    with open(possession_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["player_id", "team", "start_frame", "end_frame", "duration_s"])
        for s in spells:
            writer.writerow([s.player_id, s.team, s.start, s.end, f"{(s.end - s.start + 1) / fps:.2f}"])

    with open(config.paths.events, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "frame", "time_s", "from_player", "to_player", "from_team", "to_team"])
        for e in events:
            writer.writerow([
                e.kind, e.frame, f"{e.frame / fps:.2f}", e.from_player, e.to_player, e.from_team, e.to_team,
            ])

    shares = team_possession(spells)
    kinds = Counter((e.kind, e.from_team) for e in events)

    print("Possession Analysis")
    print("===================")
    print(f"Ball tracked in {len(ball)} frames | someone on the ball in {len(owners)} frames")
    print()
    for team, share in shares.items():
        print(f"{team}: {share:5.0%} possession | passes {kinds[('pass', team)]:2d} | "
              f"balls lost {kinds[('turnover', team)]:2d}")

    print()
    print("Sequence:")
    for s in spells:
        print(f"  {s.start / fps:5.1f}s - {s.end / fps:5.1f}s  player {s.player_id:2d} ({s.team})")

    network = passing_network(events)
    if network:
        print()
        print("Passing links:", ", ".join(f"{a}->{b} x{n}" for (a, b), n in sorted(network.items())))

    print()
    print(f"Saved: {possession_path}")
    print(f"Saved: {config.paths.events}")


if __name__ == "__main__":
    main()
