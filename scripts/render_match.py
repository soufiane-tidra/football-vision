"""Render the annotated match video from the analysed data.

Shows, on every frame: players colored by team with id and live speed,
goalkeepers and referees, the ball with a trail, who is in possession, a
running possession bar and a 2D minimap.

Requires:
    python -m scripts.build_players        -> players.csv
    python -m scripts.track_ball           -> ball.csv
    python -m scripts.analyze_possession   -> possession.csv

Run:
    python -m scripts.render_match

Output (outputs/match/):
    annotated.mp4     full annotated video
    annotated.gif     short excerpt for the README
    heatmaps.png      team and ball heatmaps
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.analytics.movement import motion_profile
from src.ball.ball_io import load_ball
from src.classification.team_classifier import jersey_color
from src.pitch.drawing import PitchDiagram
from src.pitch.landmarks import PitchDimensions
from src.tracking.players_io import load_players
from src.utils.config import load_config
from src.video.video import get_video_info
from src.visualization.annotator import (
    display_color,
    draw_ball,
    draw_heatmap,
    draw_minimap,
    draw_player,
    draw_possession_bar,
)


TEAMS = ("team_1", "team_2")
REFEREE_COLOR = (0, 230, 255)
BALL_COLOR = (255, 255, 255)
TRAIL_FRAMES = 20
BANNER = "FootballVision  |  detection + tracking  |  teams and roles  |  ball and possession  |  pixels -> meters"


def read_video(video_path):
    video = cv2.VideoCapture(str(video_path))
    while True:
        success, frame = video.read()
        if not success:
            break
        yield frame
    video.release()


def team_colors(video_path, players, info, samples_per_team=60):
    """Display color of each team, taken from the jerseys actually seen in the video."""

    wanted = {}
    for player_id, track in players.items():
        details = info[player_id]
        if details.role != "player" or details.team not in TEAMS:
            continue
        for frame, box in list(zip(track.frames, track.boxes))[::25]:
            wanted.setdefault(frame, []).append((details.team, box))

    samples = {team: [] for team in TEAMS}

    for frame_number, frame in enumerate(read_video(video_path)):
        for team, box in wanted.get(frame_number, []):
            if len(samples[team]) < samples_per_team:
                color = jersey_color(frame, box)
                if color is not None:
                    samples[team].append(color)
        if all(len(found) >= samples_per_team for found in samples.values()):
            break

    fallback = {"team_1": (255, 160, 60), "team_2": (60, 60, 255)}
    return {
        team: display_color(np.median(found, axis=0)) if found else fallback[team]
        for team, found in samples.items()
    }


def load_spells(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as file:
        return [
            (int(r["player_id"]), r["team"], int(r["start_frame"]), int(r["end_frame"]))
            for r in csv.DictReader(file)
        ]


def draw_banner(frame):
    cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.putText(
        frame, BANNER, (20, frame.shape[0] - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
    )


def main():
    config = load_config()
    settings = config.render
    dims = PitchDimensions(length=config.pitch.length, width=config.pitch.width)
    video_info = get_video_info(config.video)
    fps, width, height = video_info["fps"], video_info["width"], video_info["height"]

    output_dir = Path(config.paths.outputs) / "match"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1/4 Loading players, ball and possession...")
    players, info = load_players(config.paths.players)
    ball = load_ball(config.paths.ball) if Path(config.paths.ball).exists() else {}
    spells = load_spells(config.paths.possession)

    motion = {player_id: motion_profile(track, fps) for player_id, track in players.items()}

    boxes_by_frame = {}
    for player_id, track in players.items():
        for frame, box in zip(track.frames, track.boxes):
            boxes_by_frame.setdefault(frame, []).append((player_id, box))

    owner_at = {}
    possession_flags = {team: np.zeros(video_info["frame_count"] + 1) for team in TEAMS}
    for player_id, team, start, end in spells:
        for frame in range(start, end + 1):
            owner_at[frame] = player_id
        if team in possession_flags:
            possession_flags[team][start:end + 1] = 1
    cumulative = {team: np.cumsum(flags) for team, flags in possession_flags.items()}

    print("2/4 Reading team colors from the video...")
    colors = team_colors(config.video, players, info)

    def color_of(player_id):
        details = info[player_id]
        return REFEREE_COLOR if details.role == "referee" else colors.get(details.team, REFEREE_COLOR)

    print("3/4 Rendering...")
    writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    minimap = PitchDiagram(dims, scale=3.6, margin=3.0)
    minimap_origin = (width - minimap.image.shape[1] - 20, 20)
    bar_origin = (minimap_origin[0], minimap_origin[1] + minimap.image.shape[0] + 10)
    bar_size = (minimap.image.shape[1], 26)

    gif_start = int(settings.gif_start_s * fps)
    gif_end = gif_start + int(settings.gif_duration_s * fps)
    gif_step = max(1, int(round(fps / settings.gif_fps)))
    gif_size = (settings.gif_width, int(settings.gif_width * height / width))
    gif_frames = []

    for frame_number, frame in enumerate(read_video(config.video)):
        points, point_colors = [], []
        owner = owner_at.get(frame_number)

        for player_id, box in boxes_by_frame.get(frame_number, []):
            positions, speeds = motion[player_id]
            color = color_of(player_id)
            details = info[player_id]

            if details.role == "referee":
                label = "REF"
            else:
                prefix = "GK " if details.role == "goalkeeper" else ""
                speed = speeds.get(frame_number)
                label = f"{prefix}{player_id}" + (f" | {speed:.0f} km/h" if speed is not None else "")

            draw_player(frame, box, color, label)

            if player_id == owner:
                x1, _, x2, y2 = (int(v) for v in box)
                axes = ((x2 - x1) // 2 + 12, max((x2 - x1) // 5, 4) + 6)
                cv2.ellipse(frame, ((x1 + x2) // 2, y2), axes, 0, 0, 360, BALL_COLOR, 2, cv2.LINE_AA)

            if frame_number in positions and details.role != "referee":
                points.append(positions[frame_number])
                point_colors.append(color)

        if frame_number in ball:
            trail = [
                (ball[f]["x"], ball[f]["y"])
                for f in range(frame_number - TRAIL_FRAMES, frame_number + 1) if f in ball
            ]
            draw_ball(frame, (ball[frame_number]["x"], ball[frame_number]["y"]), trail, BALL_COLOR)

            if ball[frame_number]["pitch_x"] is not None:
                points.append((ball[frame_number]["pitch_x"], ball[frame_number]["pitch_y"]))
                point_colors.append(BALL_COLOR)

        draw_minimap(frame, minimap, points, point_colors, minimap_origin)
        draw_possession_bar(
            frame, [float(cumulative[team][frame_number]) for team in TEAMS],
            [colors[team] for team in TEAMS], bar_origin, bar_size,
        )
        draw_banner(frame)
        writer.write(frame)

        if gif_start <= frame_number < gif_end and (frame_number - gif_start) % gif_step == 0:
            small = cv2.resize(frame, gif_size, interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

    writer.release()

    if gif_frames:
        gif_frames[0].save(
            output_dir / "annotated.gif", save_all=True, append_images=gif_frames[1:],
            duration=int(1000 / settings.gif_fps), loop=0, optimize=True,
        )

    print("4/4 Heatmaps...")
    diagram = PitchDiagram(dims, scale=6)
    panels = []

    for team in TEAMS:
        positions = [
            position for player_id in players
            if info[player_id].team == team and info[player_id].role == "player"
            for position in motion[player_id][0].values()
        ]
        panel = draw_heatmap(diagram, positions)
        title = team.replace("_", " ").title()
        cv2.putText(panel, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[team], 2)
        panels.append(panel)

    ball_positions = [(b["pitch_x"], b["pitch_y"]) for b in ball.values() if b["pitch_x"] is not None]
    panel = draw_heatmap(diagram, ball_positions, sigma_m=2.0)
    cv2.putText(panel, "Ball", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BALL_COLOR, 2)
    panels.append(panel)

    cv2.imwrite(str(output_dir / "heatmaps.png"), np.hstack(panels))

    print()
    for name in ("annotated.mp4", "annotated.gif", "heatmaps.png"):
        print(f"Saved: {output_dir / name}")


if __name__ == "__main__":
    main()
