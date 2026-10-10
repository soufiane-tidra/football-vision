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


# Speed zones commonly used in football tracking data (km/h).
HIGH_SPEED_KMH = 20.0
SPRINT_KMH = 25.0


@dataclass
class MovementMetrics:
    track_id: int
    time_on_pitch_s: float
    distance_m: float
    avg_speed_kmh: float
    max_speed_mps: float
    max_speed_kmh: float
    high_speed_distance_m: float = 0.0   # distance covered above HIGH_SPEED_KMH
    sprint_distance_m: float = 0.0       # distance covered above SPRINT_KMH
    sprints: int = 0                     # efforts above SPRINT_KMH lasting at least min_sprint_s


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


def sustained_max_speed(speeds, valid, window):
    """Highest speed held over `window` consecutive valid steps (their mean).

    A top speed must be sustained: a single fast frame-to-frame step is
    measurement noise, not a sprint.
    """

    if window <= 1:
        return float(speeds[valid].max()) if valid.any() else 0.0

    if len(speeds) < window:
        return 0.0

    kernel = np.ones(window)
    all_valid = np.convolve(valid.astype(float), kernel, mode="valid") == window

    if not all_valid.any():
        return 0.0

    means = np.convolve(np.where(valid, speeds, 0.0), kernel, mode="valid") / window

    return float(means[all_valid].max())


def rolling_mean(values, window):
    """Centered moving average that keeps the length (edges use the available samples)."""

    if window <= 1 or len(values) == 0:
        return np.asarray(values, dtype=float)

    kernel = np.ones(min(window, len(values)))
    sums = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones(len(values)), kernel, mode="same")

    return sums / counts


def count_efforts(above, min_frames):
    """Number of runs of consecutive True values lasting at least min_frames."""

    count = run = 0
    for flag in above:
        run = run + 1 if flag else 0
        if run == min_frames:
            count += 1

    return count


def compute_movement_metrics(
    player,
    fps,
    smoothing_s=0.4,
    max_speed_mps=12.0,
    min_segment_s=0.5,
    sustain_s=0.5,
    min_sprint_s=1.0,
):
    """Distance and speed in real units for one track.

    max_speed_mps: faster steps are physically impossible for a player
    (world record ~12.4 m/s) and come from tracking errors; they are ignored.
    sustain_s: the top speed is the fastest speed held for this long; speed
    zones are also decided on speed averaged over this window.
    """

    window = max(1, int(round(smoothing_s * fps)))
    min_frames = int(min_segment_s * fps)
    sustain = max(1, int(round(sustain_s * fps)))
    min_sprint_frames = max(1, int(round(min_sprint_s * fps)))

    distance = 0.0
    duration = 0.0
    max_speed = 0.0
    high_speed_distance = 0.0
    sprint_distance = 0.0
    sprints = 0

    for frames, xy in pitch_segments(player):
        if len(frames) < max(min_frames, 2):
            continue

        speeds = calculate_speeds_mps(smooth_positions(xy, window), fps)
        valid = speeds <= max_speed_mps
        steps = np.where(valid, speeds, 0.0) / fps

        distance += float(steps.sum())
        duration += float(valid.sum() / fps)
        max_speed = max(max_speed, sustained_max_speed(speeds, valid, sustain))

        sustained_kmh = rolling_mean(np.where(valid, speeds, 0.0), sustain) * MPS_TO_KMH
        high_speed_distance += float(steps[sustained_kmh >= HIGH_SPEED_KMH].sum())
        sprint_distance += float(steps[sustained_kmh >= SPRINT_KMH].sum())
        sprints += count_efforts(sustained_kmh >= SPRINT_KMH, min_sprint_frames)

    avg_speed = distance / duration if duration > 0 else 0.0

    return MovementMetrics(
        track_id=player.track_id,
        time_on_pitch_s=duration,
        distance_m=distance,
        avg_speed_kmh=avg_speed * MPS_TO_KMH,
        max_speed_mps=max_speed,
        max_speed_kmh=max_speed * MPS_TO_KMH,
        high_speed_distance_m=high_speed_distance,
        sprint_distance_m=sprint_distance,
        sprints=sprints,
    )


def motion_profile(player, fps, smoothing_s=0.5):
    """Per-frame smoothed pitch position and speed for one track.

    Returns ({frame: (x, y)}, {frame: km/h}); frames without a valid position are absent.
    """

    window = max(1, int(round(smoothing_s * fps)))
    positions, speeds = {}, {}

    for frames, xy in pitch_segments(player):
        if len(frames) < 3:
            continue

        smooth = smooth_positions(xy, window)
        kmh = rolling_mean(calculate_speeds_mps(smooth, fps), window) * MPS_TO_KMH

        for i, frame in enumerate(frames):
            positions[int(frame)] = (float(smooth[i, 0]), float(smooth[i, 1]))
            speeds[int(frame)] = float(kmh[max(i - 1, 0)])

    return positions, speeds
