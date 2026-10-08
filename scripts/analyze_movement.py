"""Real-world player movement: distance (m) and speed (km/h).

Pipeline:
    tracks.csv (pixels) --per-frame homography--> pitch meters --> metrics

Requires:
    python -m scripts.track                  -> data/processed/tracks.csv
    python -m scripts.compute_homographies   -> data/processed/homographies.npy

Run:
    python -m scripts.analyze_movement

Output:
    data/processed/tracks_pitch.csv     every detection with pitch_x / pitch_y (m)
    data/processed/player_metrics.csv   one row per track
"""

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.analytics.movement import compute_movement_metrics
from src.pitch.calibration import Calibration
from src.pitch.camera_motion import load_homographies
from src.pitch.projection import on_pitch_ratio, project_tracks_to_pitch
from src.tracking.loader import load_player_tracks
from src.utils.config import load_config
from src.video.video import get_video_info


def save_tracks_pitch(players, path):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "track_id", "image_x", "image_y", "pitch_x", "pitch_y"])

        for player in players.values():
            for frame, (ix, iy), (px, py) in zip(player.frames, player.positions, player.pitch_positions):
                writer.writerow([
                    frame, player.track_id, f"{ix:.1f}", f"{iy:.1f}",
                    "" if np.isnan(px) else f"{px:.2f}",
                    "" if np.isnan(py) else f"{py:.2f}",
                ])


def main():
    config = load_config()
    settings = config.movement

    tracks_pitch_path = Path(config.paths.tracks_pitch)
    metrics_path = Path(config.paths.player_metrics)

    calibration = Calibration.load(config.paths.calibration)
    fps = get_video_info(calibration.video_path)["fps"]
    homographies = load_homographies(config.paths.homographies)

    players = load_player_tracks(config.paths.tracks, use_feet=True)
    project_tracks_to_pitch(players, homographies, calibration.pitch)
    save_tracks_pitch(players, tracks_pitch_path)

    metrics = []
    skipped = 0

    for player in players.values():
        if on_pitch_ratio(player) < settings.min_on_pitch_ratio:
            skipped += 1   # mostly off the pitch: staff, cameramen, crowd
            continue

        m = compute_movement_metrics(
            player, fps,
            smoothing_s=settings.smoothing_s,
            max_speed_mps=settings.max_speed_mps,
        )

        if m.time_on_pitch_s < settings.min_time_on_pitch_s:
            skipped += 1
            continue

        metrics.append(m)

    metrics.sort(key=lambda m: m.distance_m, reverse=True)

    print("Player Movement Analysis (real-world units)")
    print("===========================================")
    print(f"Tracks: {len(players)} | reported: {len(metrics)} | skipped (off-pitch/short): {skipped}")
    print()
    print(f"{'track':>5} {'time (s)':>9} {'dist (m)':>9} {'avg km/h':>9} {'max km/h':>9}")

    for m in metrics:
        print(
            f"{m.track_id:5d} {m.time_on_pitch_s:9.1f} {m.distance_m:9.1f} "
            f"{m.avg_speed_kmh:9.1f} {m.max_speed_kmh:9.1f}"
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(metrics[0]).keys()) if metrics else ["track_id"])
        writer.writeheader()
        for m in metrics:
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(m).items()})

    print()
    print(f"Saved: {tracks_pitch_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
