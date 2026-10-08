import math
from dataclasses import dataclass

import numpy as np


MPS_TO_KMH = 3.6


def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def calculate_total_distance(player):
    total_distance = 0.0

    for i in range(1, len(player.positions)):
        total_distance += calculate_distance(
            player.positions[i - 1],
            player.positions[i]
        )

    return total_distance


def calculate_speeds(player, fps):
    speeds = []

    for i in range(1, len(player.positions)):

        distance = calculate_distance(
            player.positions[i - 1],
            player.positions[i]
        )

        frame_difference = (
            player.frames[i] -
            player.frames[i - 1]
        )

        if frame_difference <= 0:
            speeds.append(0.0)
            continue

        time_seconds = frame_difference / fps

        speed = distance / time_seconds

        speeds.append(speed)

    return speeds


def calculate_max_speed(player, fps):
    speeds = calculate_speeds(player, fps)

    if not speeds:
        return 0.0

    return max(speeds)


# ---------------------------------------------------------------------------
# Real-world movement (meters), from player.pitch_positions
# ---------------------------------------------------------------------------


@dataclass
class MovementMetrics:
    track_id: int
    time_on_pitch_s: float
    distance_m: float
    avg_speed_kmh: float
    max_speed_mps: float
    max_speed_kmh: float


def pitch_segments(player, max_gap_frames=5):
    """Split a track into continuous segments of pitch positions.

    Missing frames (NaN / not detected) up to max_gap_frames are filled by
    linear interpolation; longer gaps start a new segment.
    Returns a list of (frames, xy) with one row per frame.
    """

    frames = np.asarray(player.frames)
    xy = np.asarray(player.pitch_positions, dtype=float).reshape(-1, 2)

    valid = ~np.isnan(xy).any(axis=1)
    frames, xy = frames[valid], xy[valid]

    if len(frames) == 0:
        return []

    breaks = np.where(np.diff(frames) > max_gap_frames)[0] + 1
    segments = []

    for seg_frames, seg_xy in zip(np.split(frames, breaks), np.split(xy, breaks)):
        full = np.arange(seg_frames[0], seg_frames[-1] + 1)
        filled = np.column_stack([
            np.interp(full, seg_frames, seg_xy[:, 0]),
            np.interp(full, seg_frames, seg_xy[:, 1]),
        ])
        segments.append((full, filled))

    return segments


def smooth_positions(xy, window):
    """Centered moving average; removes detection jitter before differentiating."""

    if window <= 1 or len(xy) < 3:
        return xy

    # Odd window, at most the segment length.
    half = min(window // 2, (len(xy) - 1) // 2)
    if half == 0:
        return xy

    # Point-reflect the ends (2*x0 - x[k]): a plain moving average would pull
    # the first/last positions inwards and under-estimate the distance.
    left = 2 * xy[0] - xy[half:0:-1]
    right = 2 * xy[-1] - xy[-2:-half - 2:-1]
    padded = np.vstack([left, xy, right])

    kernel = np.ones(2 * half + 1) / (2 * half + 1)

    return np.column_stack([
        np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)
    ])


def calculate_speeds_mps(xy, fps):
    """Speed (m/s) between consecutive frames of a smoothed segment."""

    return np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps


def compute_movement_metrics(
    player,
    fps,
    smoothing_s=0.4,
    max_speed_mps=12.0,
    min_segment_s=0.5,
):
    """Distance and speed in real units for one track.

    max_speed_mps: faster steps are physically impossible for a player
    (world record ~12.4 m/s) and come from tracking errors; they are ignored.
    """

    window = max(1, int(round(smoothing_s * fps)))
    min_frames = int(min_segment_s * fps)

    distance = 0.0
    duration = 0.0
    max_speed = 0.0

    for frames, xy in pitch_segments(player):
        if len(frames) < max(min_frames, 2):
            continue

        speeds = calculate_speeds_mps(smooth_positions(xy, window), fps)
        valid = speeds <= max_speed_mps

        distance += float(speeds[valid].sum() / fps)
        duration += float(valid.sum() / fps)

        if valid.any():
            max_speed = max(max_speed, float(speeds[valid].max()))

    avg_speed = distance / duration if duration > 0 else 0.0

    return MovementMetrics(
        track_id=player.track_id,
        time_on_pitch_s=duration,
        distance_m=distance,
        avg_speed_kmh=avg_speed * MPS_TO_KMH,
        max_speed_mps=max_speed,
        max_speed_kmh=max_speed * MPS_TO_KMH,
    )
