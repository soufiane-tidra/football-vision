"""Team shape: how a team occupies the pitch, frame by frame.

For the outfield players of one team in one frame:
    centroid          mean position
    width             extent across the pitch (y), in meters
    depth             extent along the pitch (x), in meters
    area              area of the convex hull of the players, in m2 (compactness)
    line_height       distance from the team's own goal line to its defensive
                      line (second deepest outfield player, as for offside)

A panning camera rarely shows all ten outfield players, so a frame is only
measured when at least min_players are visible; values describe the visible
block of the team.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TeamShape:
    frame: int
    players: int
    centroid_x: float
    centroid_y: float
    width: float
    depth: float
    area: float
    line_height: float


def shape_of(positions, frame=0, defends_left=True, pitch_length=105.0):
    """TeamShape for one set of (x, y) positions in meters."""

    points = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    xs = np.sort(points[:, 0])

    hull = cv2.convexHull(points) if len(points) >= 3 else None
    area = float(cv2.contourArea(hull)) if hull is not None else 0.0

    if len(xs) >= 2:
        line_height = float(xs[1]) if defends_left else float(pitch_length - xs[-2])
    else:
        line_height = float("nan")

    return TeamShape(
        frame=frame,
        players=len(points),
        centroid_x=float(points[:, 0].mean()),
        centroid_y=float(points[:, 1].mean()),
        width=float(np.ptp(points[:, 1])),
        depth=float(np.ptp(points[:, 0])),
        area=area,
        line_height=line_height,
    )


def defending_left(positions_by_team):
    """Which team defends the left goal: the one whose deepest players are closest to it.

    positions_by_team: {team: iterable of (x, y)} over the whole sequence.
    """

    depth = {}
    for team, positions in positions_by_team.items():
        xs = np.asarray(list(positions), dtype=float).reshape(-1, 2)[:, 0]
        depth[team] = float(np.percentile(xs, 5)) if len(xs) else float("inf")

    return min(depth, key=depth.get)


def team_shapes(positions_by_frame, defends_left=True, min_players=7, pitch_length=105.0):
    """positions_by_frame: {frame: [(x, y), ...]} for one team's outfield players.

    Returns a list of TeamShape for the frames with enough visible players.
    """

    return [
        shape_of(positions, frame, defends_left, pitch_length)
        for frame, positions in sorted(positions_by_frame.items())
        if len(positions) >= min_players
    ]


def summarize(shapes):
    """Mean of each metric over a list of TeamShape (empty dict if no frames)."""

    if not shapes:
        return {}

    return {
        name: float(np.nanmean([getattr(shape, name) for shape in shapes]))
        for name in ("players", "centroid_x", "centroid_y", "width", "depth", "area", "line_height")
    }
