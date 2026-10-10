"""Matplotlib charts for the match report. Every function returns PNG bytes."""

import io

import matplotlib.pyplot as plt
import numpy as np

from src.pitch.landmarks import PitchDimensions, get_pitch_lines


# Charts are written to files, never shown: no display is needed.
plt.switch_backend("Agg")

GRASS = "#2f7a3d"
LINE = "#ffffff"


def _rgb(bgr):
    blue, green, red = bgr
    return (red / 255, green / 255, blue / 255)


def _to_png(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return buffer.getvalue()


def draw_pitch(axis, dims=PitchDimensions()):
    """Draw the pitch markings on a matplotlib axis (meters, y pointing down like the camera view)."""

    axis.set_facecolor(GRASS)
    for line in get_pitch_lines(dims):
        axis.plot(line[:, 0], line[:, 1], color=LINE, linewidth=1.2)

    axis.set_xlim(-3, dims.length + 3)
    axis.set_ylim(dims.width + 3, -3)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def passing_network_chart(nodes, links, team_colors, dims=PitchDimensions()):
    """nodes: {player_id: (x, y, team)}; links: {(from_id, to_id): passes}; team_colors: {team: BGR}."""

    figure, axis = plt.subplots(figsize=(9, 6), facecolor="white")
    draw_pitch(axis, dims)

    for (source, target), count in links.items():
        if source in nodes and target in nodes:
            x1, y1, team = nodes[source]
            x2, y2, _ = nodes[target]
            axis.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={
                    "arrowstyle": "-|>", "color": "white", "alpha": 0.9,
                    "linewidth": 1.5 + 1.5 * count, "shrinkA": 9, "shrinkB": 9,
                },
            )

    for player_id, (x, y, team) in nodes.items():
        axis.scatter([x], [y], s=330, color=_rgb(team_colors.get(team, (200, 200, 200))),
                     edgecolors="black", linewidths=1.2, zorder=3)
        axis.text(x, y, str(player_id), ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    axis.set_title("Passing network (players at their average position)")

    return _to_png(figure)


def team_shape_chart(series, team_colors, fps):
    """series: {team: list of TeamShape}. Width, depth and area over time."""

    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, facecolor="white")
    metrics = (("width", "Width (m)"), ("depth", "Depth (m)"), ("area", "Area (m2)"))

    for axis, (name, label) in zip(axes, metrics):
        for team, shapes in series.items():
            if not shapes:
                continue
            times = np.array([shape.frame for shape in shapes]) / fps
            values = np.array([getattr(shape, name) for shape in shapes])
            axis.plot(times, values, color=_rgb(team_colors.get(team, (120, 120, 120))),
                      linewidth=1.8, label=team.replace("_", " ").title())
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[0].set_title("Team shape over time (visible outfield players)")
    axes[-1].set_xlabel("Time (s)")

    return _to_png(figure)
