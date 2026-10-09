"""The 32 pitch keypoints of the Roboflow "football-field-detection" dataset.

Same order as roboflow/sports SoccerPitchConfiguration.vertices, but placed
on *our* pitch (meters, real penalty-box / goal-area sizes). Index i here is
keypoint i of the model output (vertex i + 1 in the reference).
"""

import numpy as np

from src.pitch.homography import PitchMapper
from src.pitch.landmarks import PitchDimensions


KEYPOINT_NAMES = [
    "corner_top_left",                    # 1
    "left_penalty_goalline_top",          # 2
    "left_goal_area_goalline_top",        # 3
    "left_goal_area_goalline_bottom",     # 4
    "left_penalty_goalline_bottom",       # 5
    "corner_bottom_left",                 # 6
    "left_goal_area_corner_top",          # 7
    "left_goal_area_corner_bottom",       # 8
    "left_penalty_spot",                  # 9
    "left_penalty_corner_top",            # 10
    "left_penalty_line_goal_area_top",    # 11  on the box line, level with the goal area
    "left_penalty_line_goal_area_bottom", # 12
    "left_penalty_corner_bottom",         # 13
    "halfway_top",                        # 14
    "center_circle_top",                  # 15
    "center_circle_bottom",               # 16
    "halfway_bottom",                     # 17
    "right_penalty_corner_top",           # 18
    "right_penalty_line_goal_area_top",   # 19
    "right_penalty_line_goal_area_bottom",# 20
    "right_penalty_corner_bottom",        # 21
    "right_penalty_spot",                 # 22
    "right_goal_area_corner_top",         # 23
    "right_goal_area_corner_bottom",      # 24
    "corner_top_right",                   # 25
    "right_penalty_goalline_top",         # 26
    "right_goal_area_goalline_top",       # 27
    "right_goal_area_goalline_bottom",    # 28
    "right_penalty_goalline_bottom",      # 29
    "corner_bottom_right",                # 30
    "center_circle_left",                 # 31
    "center_circle_right",                # 32
]

NUM_KEYPOINTS = len(KEYPOINT_NAMES)


def pitch_keypoints(dims: PitchDimensions = PitchDimensions()) -> np.ndarray:
    """(32, 2) pitch coordinates in meters, in model keypoint order."""

    L, W = dims.length, dims.width
    cy = W / 2
    pa_top, pa_bottom = cy - dims.penalty_area_width / 2, cy + dims.penalty_area_width / 2
    ga_top, ga_bottom = cy - dims.goal_area_width / 2, cy + dims.goal_area_width / 2
    pa, ga = dims.penalty_area_depth, dims.goal_area_depth
    spot, r = dims.penalty_spot_distance, dims.center_circle_radius

    return np.array([
        (0, 0), (0, pa_top), (0, ga_top), (0, ga_bottom), (0, pa_bottom), (0, W),
        (ga, ga_top), (ga, ga_bottom),
        (spot, cy),
        (pa, pa_top), (pa, ga_top), (pa, ga_bottom), (pa, pa_bottom),
        (L / 2, 0), (L / 2, cy - r), (L / 2, cy + r), (L / 2, W),
        (L - pa, pa_top), (L - pa, ga_top), (L - pa, ga_bottom), (L - pa, pa_bottom),
        (L - spot, cy),
        (L - ga, ga_top), (L - ga, ga_bottom),
        (L, 0), (L, pa_top), (L, ga_top), (L, ga_bottom), (L, pa_bottom), (L, W),
        (L / 2 - r, cy), (L / 2 + r, cy),
    ], dtype=float)


def homography_from_keypoints(
    image_xy,
    confidence,
    pitch_xy,
    min_confidence=0.5,
    min_points=4,
    max_error_m=1.0,
    min_spread_m=3.0,
):
    """Build a PitchMapper from detected keypoints, or None if the frame is unreliable.

    Rejected when: too few confident points, points (nearly) on one line
    (e.g. only the halfway line visible), or inliers disagreeing by more
    than max_error_m.
    """

    image_xy = np.asarray(image_xy, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    pitch_xy = np.asarray(pitch_xy, dtype=float)

    keep = confidence >= min_confidence
    if keep.sum() < min_points:
        return None

    image_pts, pitch_pts = image_xy[keep], pitch_xy[keep]

    # Spread in the weakest direction: ~0 when all points are collinear.
    centered = pitch_pts - pitch_pts.mean(axis=0)
    if np.linalg.svd(centered, compute_uv=False)[-1] / np.sqrt(len(pitch_pts)) < min_spread_m:
        return None

    try:
        mapper = PitchMapper(image_pts, pitch_pts, ransac_threshold_m=max_error_m)
    except ValueError:
        return None

    inliers = mapper.inlier_mask
    if inliers.sum() < min_points:
        return None

    if mapper.reprojection_errors()[inliers].mean() > max_error_m:
        return None

    return mapper


def smooth_homographies(homographies, valid, window):
    """Temporal moving average of valid homographies (removes frame-to-frame jitter).

    Frames whose window has no valid homography are returned unchanged.
    """

    homographies = np.asarray(homographies, dtype=float)
    valid = np.asarray(valid, dtype=bool)

    if window <= 1:
        return homographies.copy()

    normalized = homographies / np.linalg.norm(homographies, axis=(1, 2), keepdims=True)
    half = window // 2
    smoothed = homographies.copy()

    for t in range(len(homographies)):
        lo, hi = max(0, t - half), min(len(homographies), t + half + 1)
        neighbours = [i for i in range(lo, hi) if valid[i]]
        if not neighbours:
            continue

        reference = normalized[t] if valid[t] else normalized[neighbours[0]]
        total = np.zeros((3, 3))
        for i in neighbours:
            sign = 1.0 if np.sum(normalized[i] * reference) >= 0 else -1.0
            total += sign * normalized[i]

        smoothed[t] = total / len(neighbours)

    return smoothed
