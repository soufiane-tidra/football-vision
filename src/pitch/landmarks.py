"""Standard football pitch model: dimensions, named landmarks and line geometry.

Coordinate system (meters):

    (0, 0) ------------------- (length, 0)      <- far touchline
      |                             |
    left goal                   right goal
      |                             |
    (0, width) --------------- (length, width)  <- near touchline (camera side)

X runs along the touchlines, Y runs along the goal lines.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchDimensions:
    length: float = 105.0
    width: float = 68.0

    # Fixed by the Laws of the Game.
    penalty_area_depth: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    penalty_spot_distance: float = 11.0
    center_circle_radius: float = 9.15
    goal_width: float = 7.32

    @property
    def center(self) -> tuple[float, float]:
        return self.length / 2, self.width / 2


def get_landmarks(dims: PitchDimensions = PitchDimensions()) -> dict[str, tuple[float, float]]:
    """Return named, well-defined pitch points (line intersections, spots)."""

    L, W = dims.length, dims.width
    cx, cy = dims.center
    r = dims.center_circle_radius

    pa_half = dims.penalty_area_width / 2
    ga_half = dims.goal_area_width / 2
    goal_half = dims.goal_width / 2

    # Where the penalty arc meets the penalty area line.
    arc_dx = dims.penalty_area_depth - dims.penalty_spot_distance
    arc_half = float(np.sqrt(r ** 2 - arc_dx ** 2))

    landmarks = {
        # Corners
        "corner_top_left": (0.0, 0.0),
        "corner_bottom_left": (0.0, W),
        "corner_top_right": (L, 0.0),
        "corner_bottom_right": (L, W),

        # Halfway line and center circle
        "halfway_top": (cx, 0.0),
        "halfway_bottom": (cx, W),
        "center_spot": (cx, cy),
        "center_circle_top": (cx, cy - r),
        "center_circle_bottom": (cx, cy + r),
        "center_circle_left": (cx - r, cy),
        "center_circle_right": (cx + r, cy),
    }

    for side, x_goal, direction in (("left", 0.0, 1.0), ("right", L, -1.0)):
        x_pa = x_goal + direction * dims.penalty_area_depth
        x_ga = x_goal + direction * dims.goal_area_depth
        x_spot = x_goal + direction * dims.penalty_spot_distance

        landmarks.update({
            f"{side}_penalty_goalline_top": (x_goal, cy - pa_half),
            f"{side}_penalty_goalline_bottom": (x_goal, cy + pa_half),
            f"{side}_penalty_corner_top": (x_pa, cy - pa_half),
            f"{side}_penalty_corner_bottom": (x_pa, cy + pa_half),
            f"{side}_penalty_arc_top": (x_pa, cy - arc_half),
            f"{side}_penalty_arc_bottom": (x_pa, cy + arc_half),
            f"{side}_penalty_spot": (x_spot, cy),
            f"{side}_goal_area_goalline_top": (x_goal, cy - ga_half),
            f"{side}_goal_area_goalline_bottom": (x_goal, cy + ga_half),
            f"{side}_goal_area_corner_top": (x_ga, cy - ga_half),
            f"{side}_goal_area_corner_bottom": (x_ga, cy + ga_half),
            f"{side}_goal_post_top": (x_goal, cy - goal_half),
            f"{side}_goal_post_bottom": (x_goal, cy + goal_half),
        })

    return landmarks


def _arc(cx, cy, r, start_deg, end_deg, n=40):
    angles = np.radians(np.linspace(start_deg, end_deg, n))
    return np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)


def get_pitch_lines(dims: PitchDimensions = PitchDimensions()) -> list[np.ndarray]:
    """Return the pitch markings as polylines in meters (each an (N, 2) array)."""

    L, W = dims.length, dims.width
    cx, cy = dims.center
    r = dims.center_circle_radius

    lines = [
        np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], dtype=float),
        np.array([[cx, 0], [cx, W]], dtype=float),
        _arc(cx, cy, r, 0, 360, n=100),
    ]

    pa_half = dims.penalty_area_width / 2
    ga_half = dims.goal_area_width / 2
    arc_dx = dims.penalty_area_depth - dims.penalty_spot_distance
    arc_angle = np.degrees(np.arccos(arc_dx / r))

    for x_goal, d in ((0.0, 1.0), (L, -1.0)):
        x_pa = x_goal + d * dims.penalty_area_depth
        x_ga = x_goal + d * dims.goal_area_depth
        x_spot = x_goal + d * dims.penalty_spot_distance

        lines.append(np.array([
            [x_goal, cy - pa_half], [x_pa, cy - pa_half],
            [x_pa, cy + pa_half], [x_goal, cy + pa_half],
        ], dtype=float))

        lines.append(np.array([
            [x_goal, cy - ga_half], [x_ga, cy - ga_half],
            [x_ga, cy + ga_half], [x_goal, cy + ga_half],
        ], dtype=float))

        # Penalty arc: the part of the circle around the spot outside the box.
        base = 0.0 if d > 0 else 180.0
        lines.append(_arc(x_spot, cy, r, base - arc_angle, base + arc_angle))

    return lines
