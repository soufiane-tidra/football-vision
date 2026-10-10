"""Real-world player movement: distance (m) and speed (km/h) per identified player.

Requires:
    python -m scripts.build_players   -> data/processed/players.csv

Run:
    python -m scripts.analyze_movement

Output:
    data/processed/player_metrics.csv   one row per player
"""

import csv
from dataclasses import asdict
from pathlib import Path

from src.analytics.movement import compute_movement_metrics
from src.tracking.players_io import load_players
from src.utils.config import load_config
from src.video.video import get_video_info


def main():
    config = load_config()
    settings = config.movement
    metrics_path = Path(config.paths.player_metrics)

    fps = get_video_info(config.video)["fps"]
    players, info = load_players(config.paths.players)

    rows = []

    for player_id, player in players.items():
        if info[player_id].role == "referee":
            continue

        m = compute_movement_metrics(
            player, fps,
            smoothing_s=settings.smoothing_s,
            max_speed_mps=settings.max_speed_mps,
        )

        if m.time_on_pitch_s < settings.min_time_on_pitch_s:
            continue

        row = {"player_id": player_id, "team": info[player_id].team, "role": info[player_id].role}
        row.update({k: round(v, 2) for k, v in asdict(m).items() if k != "track_id"})
        rows.append(row)

    rows.sort(key=lambda r: (r["team"], -r["distance_m"]))

    print("Player Movement Analysis (real-world units)")
    print("===========================================")
    print(f"{'id':>3} {'team':>7} {'role':>10} {'time (s)':>9} {'dist (m)':>9} {'avg km/h':>9} {'max km/h':>9}")

    for r in rows:
        print(
            f"{r['player_id']:3d} {r['team']:>7} {r['role']:>10} {r['time_on_pitch_s']:9.1f} "
            f"{r['distance_m']:9.1f} {r['avg_speed_kmh']:9.1f} {r['max_speed_kmh']:9.1f}"
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["player_id"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
