"""Build the match report: one self-contained HTML page.

Team shape metrics are computed here; everything else is read from the
files produced by the pipeline.

Requires:
    python -m scripts.run_pipeline      (or its individual steps)

Run:
    python -m scripts.build_report

Output:
    data/processed/team_shape.csv       width / depth / area / line height per frame and team
    outputs/match/report.html           open it in a browser
"""

import base64
import csv
import html
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.analytics.movement import motion_profile
from src.analytics.team_shape import defending_left, summarize, team_shapes
from src.ball.ball_io import load_ball
from src.pitch.landmarks import PitchDimensions
from src.tracking.players_io import load_players
from src.utils.config import load_config
from src.video.video import get_video_info
from src.visualization.charts import passing_network_chart, team_shape_chart


TEAMS = ("team_1", "team_2")
REPORT_COLORS = {"team_1": (230, 150, 50), "team_2": (60, 60, 230)}     # BGR, used when jerseys were not sampled


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def image_tag(png_bytes, alt):
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}">'


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    config = load_config()
    dims = PitchDimensions(length=config.pitch.length, width=config.pitch.width)
    video = get_video_info(config.video)
    fps = video["fps"]

    players, info = load_players(config.paths.players)
    ball = load_ball(config.paths.ball) if Path(config.paths.ball).exists() else {}
    spells = read_csv(config.paths.possession)
    events = read_csv(config.paths.events)
    metrics = read_csv(config.paths.player_metrics)

    # ---------- team shape ----------
    motion = {player_id: motion_profile(track, fps)[0] for player_id, track in players.items()}

    positions = {team: {} for team in TEAMS}
    for player_id, by_frame in motion.items():
        details = info[player_id]
        if details.role == "player" and details.team in positions:
            for frame, position in by_frame.items():
                positions[details.team].setdefault(frame, []).append(position)

    left_team = defending_left({
        team: [p for frame_positions in by_frame.values() for p in frame_positions]
        for team, by_frame in positions.items()
    })

    shapes = {
        team: team_shapes(positions[team], defends_left=(team == left_team), pitch_length=dims.length)
        for team in TEAMS
    }

    shape_path = Path(config.paths.team_shape)
    shape_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shape_path, "w", newline="") as file:
        writer = None
        for team in TEAMS:
            for shape in shapes[team]:
                values = {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(shape).items()}
                row = {"team": team, **values}
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)

    # ---------- numbers ----------
    possession_s = Counter()
    for spell in spells:
        possession_s[spell["team"]] += float(spell["duration_s"])
    total_possession = sum(possession_s.values())

    passes = Counter(e["from_team"] for e in events if e["kind"] == "pass")
    lost = Counter(e["from_team"] for e in events if e["kind"] == "turnover")

    identities = Counter((info[p].team or "officials", info[p].role) for p in players)
    by_team_metrics = {team: [m for m in metrics if m["team"] == team and m["role"] == "player"] for team in TEAMS}

    def total(team, column):
        return sum(float(m[column]) for m in by_team_metrics[team])

    summary_rows = []
    for team in TEAMS:
        shape = summarize(shapes[team])
        share = possession_s[team] / total_possession if total_possession else 0.0
        summary_rows.append([
            team.replace("_", " ").title(),
            f"{share:.0%}",
            passes[team],
            lost[team],
            f"{total(team, 'distance_m'):.0f}",
            f"{total(team, 'high_speed_distance_m'):.0f}",
            int(total(team, "sprints")),
            f"{shape.get('width', float('nan')):.1f}",
            f"{shape.get('depth', float('nan')):.1f}",
            f"{shape.get('area', float('nan')):.0f}",
            f"{shape.get('line_height', float('nan')):.1f}",
            "left" if team == left_team else "right",
        ])

    # ---------- charts ----------
    links = Counter((int(e["from_player"]), int(e["to_player"])) for e in events if e["kind"] == "pass")
    involved = {player for link in links for player in link}
    nodes = {}
    for player_id in involved:
        points = np.array(list(motion.get(player_id, {}).values()))
        if len(points):
            nodes[player_id] = (float(points[:, 0].mean()), float(points[:, 1].mean()), info[player_id].team)

    charts = [team_shape_chart(shapes, REPORT_COLORS, fps)]
    if nodes:
        charts.insert(0, passing_network_chart(nodes, dict(links), REPORT_COLORS, dims))

    heatmaps = Path(config.paths.outputs) / "match" / "heatmaps.png"

    player_rows = [
        [
            m["player_id"], m["team"].replace("_", " ").title(), m["role"],
            f"{float(m['time_on_pitch_s']):.1f}", f"{float(m['distance_m']):.0f}",
            f"{float(m['avg_speed_kmh']):.1f}", f"{float(m['max_speed_kmh']):.1f}",
            f"{float(m['high_speed_distance_m']):.0f}", m["sprints"],
        ]
        for m in metrics
    ]

    sequence_rows = [
        [f"{int(s['start_frame']) / fps:.1f}", f"{float(s['duration_s']):.1f}", s["player_id"],
         s["team"].replace("_", " ").title()]
        for s in spells
    ]

    # ---------- page ----------
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>FootballVision match report</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f5;color:#1c2321}",
        "main{max-width:1100px;margin:0 auto;padding:24px 16px}",
        "h1{margin:0 0 4px}h2{margin:32px 0 10px;border-bottom:2px solid #2f7a3d;padding-bottom:4px}",
        ".muted{color:#5c6763}.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}",
        ".card{background:#fff;border-radius:8px;padding:12px 16px;min-width:150px;box-shadow:0 1px 3px #0002}",
        ".card b{display:block;font-size:24px}",
        ".scroll{overflow-x:auto}table{border-collapse:collapse;background:#fff;width:100%;font-size:14px}",
        "th,td{padding:6px 10px;border-bottom:1px solid #e3e8e6;text-align:right;white-space:nowrap}",
        "th:first-child,td:first-child{text-align:left}th{background:#2f7a3d;color:#fff}",
        "img{max-width:100%;height:auto;border-radius:8px;background:#fff}",
        ".note{background:#fff8e1;border-left:4px solid #e0a800;padding:10px 14px;border-radius:4px}",
        "</style></head><body><main>",
        "<h1>FootballVision match report</h1>",
        f"<p class='muted'>{html.escape(Path(config.video).name)} · {video['duration']:.1f} s · "
        f"{video['frame_count']} frames · {video['width']}×{video['height']} · {fps:.0f} fps</p>",
        "<div class='cards'>",
        f"<div class='card'>Identified people<b>{len(players)}</b></div>",
        "<div class='card'>Outfield identities"
        f"<b>{identities[('team_1', 'player')]} + {identities[('team_2', 'player')]}</b></div>",
        f"<div class='card'>Ball tracked<b>{len(ball) / max(video['frame_count'], 1):.0%}</b>of frames</div>",
        f"<div class='card'>Passes detected<b>{sum(passes.values())}</b></div>",
        f"<div class='card'>Turnovers<b>{sum(lost.values())}</b></div>",
        "</div>",
        "<h2>Teams</h2><div class='scroll'>",
        table(
            ["Team", "Possession", "Passes", "Balls lost", "Distance (m)", "High-speed (m)", "Sprints",
             "Width (m)", "Depth (m)", "Area (m2)", "Line height (m)", "Defends"],
            summary_rows,
        ),
        "</div>",
        "<p class='muted'>Width, depth and area describe the outfield players visible in the frame "
        "(at least 7). Line height is the distance from the team's own goal line to its second deepest "
        "outfield player.</p>",
    ]

    titles = ["Passing network", "Team shape"] if nodes else ["Team shape"]
    for title, chart in zip(titles, charts):
        parts += [f"<h2>{title}</h2>", image_tag(chart, title)]

    if heatmaps.exists():
        parts += ["<h2>Heatmaps</h2>", image_tag(heatmaps.read_bytes(), "Team and ball heatmaps")]

    parts += [
        "<h2>Possession sequence</h2><div class='scroll'>",
        table(["Start (s)", "Duration (s)", "Player", "Team"], sequence_rows),
        "</div>",
        "<h2>Players</h2><div class='scroll'>",
        table(
            ["Id", "Team", "Role", "Time (s)", "Distance (m)", "Avg km/h", "Top km/h", "High-speed (m)", "Sprints"],
            player_rows,
        ),
        "</div>",
        "<h2>How to read these numbers</h2>",
        "<p class='note'>Positions come from an automatic pitch calibration that is about 2 m accurate on "
        "this footage, so distances and speeds are indicative. A player who leaves the camera view and "
        "returns gets a new id, so one real player can appear as two rows. Passes are inferred from "
        "changes of possession and have not been checked against hand-labelled events.</p>",
        "</main></body></html>",
    ]

    report_path = Path(config.paths.outputs) / "match" / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(parts), encoding="utf-8")

    print("Match Report")
    print("============")
    headers = ["team", "poss", "pass", "lost", "dist", "hsr", "spr", "width", "depth", "area", "line", "defends"]
    print("  ".join(f"{h:>7}" for h in headers))
    for row in summary_rows:
        print("  ".join(f"{str(cell):>7}" for cell in row))
    print()
    print(f"Saved: {shape_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
