"""Build the FootballVision demo from existing tracking + calibration data.

Pipeline:
    tracks.csv (YOLO + ByteTrack)
      -> per-frame pitch homography (keyframe calibration + camera tracking)
      -> player positions in meters, speed in km/h
      -> team classification from jersey colors (K-means)
      -> annotated video + 2D minimap, heatmaps, stats

Run:
    python -m scripts.make_demo

Output (outputs/demo/):
    demo.mp4            annotated video
    demo.gif            short GIF for the README
    heatmaps.png        team heatmaps
    player_stats.csv    distance / speed per player
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.analytics.movement import (
    MPS_TO_KMH,
    calculate_speeds_mps,
    compute_movement_metrics,
    pitch_segments,
    smooth_positions,
)
from src.classification.team_classifier import OTHER, TeamClassifier, jersey_color
from src.pitch.calibration import Calibration
from src.pitch.camera_motion import PitchHomographyTracker
from src.pitch.drawing import PitchDiagram
from src.pitch.projection import on_pitch_ratio, project_tracks_to_pitch
from src.tracking.player import PlayerTrack
from src.utils.config import load_config
from src.video.video import get_video_info
from src.visualization.annotator import draw_heatmap, draw_minimap, draw_player


config = load_config()

TRACKS_PATH = config.paths.tracks
CALIBRATION_PATH = config.paths.calibration
OUTPUT_DIR = Path(config.paths.outputs) / "demo"

# Segment where the pitch calibration has been visually verified.
DEMO_FRAMES = config.demo.frames

MIN_DETECTIONS = 8
SMOOTHING_S = 0.5
OTHER_COLOR = (0, 230, 255)    # referees / goalkeepers (yellow)

GIF_STEP = 4                   # 30 fps -> 7.5 fps
GIF_WIDTH = 640


def load_detections(path, max_frames):
    """frame -> list of (track_id, box) and track_id -> PlayerTrack (feet positions)."""

    detections = {}
    players = {}

    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            frame = int(row["frame"])

            if row["class_name"] != "person" or frame >= max_frames:
                continue

            track_id = int(row["track_id"])
            box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))

            detections.setdefault(frame, []).append((track_id, box))

            player = players.setdefault(track_id, PlayerTrack(track_id=track_id))
            player.add_detection(frame, (box[0] + box[2]) / 2, box[3], float(row["confidence"]))

    return detections, players


def smoothed_motion(player, fps):
    """Per-frame smoothed pitch position and speed (km/h) for one track."""

    window = max(1, int(round(SMOOTHING_S * fps)))
    positions, speeds = {}, {}

    for frames, xy in pitch_segments(player):
        if len(frames) < 3:
            continue

        smooth = smooth_positions(xy, window)
        kmh = calculate_speeds_mps(smooth, fps) * MPS_TO_KMH

        for i, frame in enumerate(frames):
            positions[int(frame)] = smooth[i]
            speeds[int(frame)] = float(kmh[max(i - 1, 0)])

    return positions, speeds


def read_frames(video_path, max_frames):
    video = cv2.VideoCapture(str(video_path))
    for _ in range(max_frames):
        success, frame = video.read()
        if not success:
            break
        yield frame
    video.release()


def classify_teams(video_path, detections, player_ids):
    samples = {}

    for frame_number, frame in enumerate(read_frames(video_path, DEMO_FRAMES)):
        if frame_number % 5:
            continue

        for track_id, box in detections.get(frame_number, []):
            if track_id in player_ids:
                color = jersey_color(frame, box)
                if color is not None:
                    samples.setdefault(track_id, []).append(color)

    track_colors = {t: np.median(c, axis=0) for t, c in samples.items() if len(c) >= 2}

    classifier = TeamClassifier()
    teams = classifier.fit_predict(track_colors)

    return teams, classifier.team_display_colors()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration = Calibration.load(CALIBRATION_PATH)
    video_path = calibration.video_path
    info = get_video_info(video_path)
    fps, width, height = info["fps"], info["width"], info["height"]

    print("1/5 Pitch homographies...")
    tracker = PitchHomographyTracker(calibration.pitch)
    homographies = tracker.process(video_path, calibration, max_frames=DEMO_FRAMES)

    print("2/5 Projecting players to the pitch...")
    detections, players = load_detections(TRACKS_PATH, DEMO_FRAMES)
    project_tracks_to_pitch(players, homographies, calibration.pitch)

    player_ids = {
        t for t, p in players.items()
        if p.detection_count >= MIN_DETECTIONS and on_pitch_ratio(p) >= 0.8
    }

    motion = {t: smoothed_motion(players[t], fps) for t in player_ids}

    print("3/5 Team classification (jersey colors)...")
    teams, team_colors = classify_teams(video_path, detections, player_ids)

    def color_of(track_id):
        team = teams.get(track_id, OTHER)
        return OTHER_COLOR if team == OTHER else team_colors[team]

    print("4/5 Rendering annotated video...")
    writer = cv2.VideoWriter(
        str(OUTPUT_DIR / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    minimap = PitchDiagram(calibration.pitch, scale=3.6, margin=3.0)
    minimap_origin = (width - minimap.image.shape[1] - 20, 20)
    gif_frames = []

    for frame_number, frame in enumerate(read_frames(video_path, DEMO_FRAMES)):
        points, colors = [], []

        for track_id, box in detections.get(frame_number, []):
            if track_id not in player_ids:
                continue

            positions, speeds = motion[track_id]
            if frame_number not in positions:
                continue

            color = color_of(track_id)
            draw_player(frame, box, color, f"{track_id} | {speeds[frame_number]:.0f} km/h")

            points.append(positions[frame_number])
            colors.append(color)

        draw_minimap(frame, minimap, points, colors, minimap_origin)
        _banner(frame)
        writer.write(frame)

        if frame_number % GIF_STEP == 0:
            small = cv2.resize(frame, (GIF_WIDTH, int(GIF_WIDTH * height / width)), interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

    writer.release()

    gif_frames[0].save(
        OUTPUT_DIR / "demo.gif", save_all=True, append_images=gif_frames[1:],
        duration=int(1000 * GIF_STEP / fps), loop=0, optimize=True,
    )

    print("5/5 Heatmaps and stats...")
    diagram = PitchDiagram(calibration.pitch, scale=6)
    panels = []

    for team in range(len(team_colors)):
        points = [
            p for t in player_ids if teams.get(t) == team
            for p in motion[t][0].values()
        ]
        panel = draw_heatmap(diagram, points)
        cv2.putText(panel, f"Team {team + 1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, team_colors[team], 2)
        panels.append(panel)

    cv2.imwrite(str(OUTPUT_DIR / "heatmaps.png"), np.hstack(panels))

    rows = []
    for t in player_ids:
        m = compute_movement_metrics(players[t], fps, smoothing_s=SMOOTHING_S)
        team = teams.get(t, OTHER)
        rows.append({
            "track_id": t,
            "team": "other" if team == OTHER else f"team_{team + 1}",
            "time_s": round(m.time_on_pitch_s, 1),
            "distance_m": round(m.distance_m, 1),
            "avg_speed_kmh": round(m.avg_speed_kmh, 1),
            "max_speed_kmh": round(m.max_speed_kmh, 1),
        })

    rows.sort(key=lambda r: r["distance_m"], reverse=True)

    with open(OUTPUT_DIR / "player_stats.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    team_counts = {name: sum(r["team"] == name for r in rows) for name in ("team_1", "team_2", "other")}

    print()
    print(f"Demo: {DEMO_FRAMES} frames ({DEMO_FRAMES / fps:.0f} s) | players: {len(rows)} | {team_counts}")
    print(f"{'track':>5} {'team':>7} {'time s':>7} {'dist m':>7} {'avg km/h':>9} {'max km/h':>9}")
    for r in rows[:10]:
        print(
            f"{r['track_id']:5d} {r['team']:>7} {r['time_s']:7.1f} {r['distance_m']:7.1f} "
            f"{r['avg_speed_kmh']:9.1f} {r['max_speed_kmh']:9.1f}"
        )

    print()
    for name in ("demo.mp4", "demo.gif", "heatmaps.png", "player_stats.csv"):
        print(f"Saved: {OUTPUT_DIR / name}")


def _banner(frame):
    text = "FootballVision  |  YOLO + ByteTrack  |  team clustering  |  homography: pixels -> meters"
    cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.putText(frame, text, (20, frame.shape[0] - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
