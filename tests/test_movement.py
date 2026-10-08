import numpy as np
import pytest

from src.analytics.movement import (
    MPS_TO_KMH,
    calculate_total_distance,
    compute_movement_metrics,
    pitch_segments,
    smooth_positions,
)
from src.tracking.player import PlayerTrack


FPS = 30.0


def make_player(frames, positions):
    player = PlayerTrack(track_id=1)
    for frame, (x, y) in zip(frames, positions):
        player.add_detection(frame, x, y, 0.9)
    player.pitch_positions = [tuple(p) for p in positions]
    return player


def straight_run(speed_mps, seconds, start=(10.0, 30.0)):
    frames = np.arange(int(seconds * FPS))
    xs = start[0] + speed_mps * frames / FPS
    return frames, np.column_stack([xs, np.full_like(xs, start[1])])


def test_constant_speed_is_measured_exactly():
    frames, positions = straight_run(speed_mps=5.0, seconds=4)
    metrics = compute_movement_metrics(make_player(frames, positions), FPS)

    expected_distance = 5.0 * (len(frames) - 1) / FPS
    assert metrics.distance_m == pytest.approx(expected_distance, rel=0.02)
    assert metrics.max_speed_kmh == pytest.approx(5.0 * MPS_TO_KMH, rel=0.02)


def test_detection_jitter_does_not_create_speed():
    rng = np.random.default_rng(0)
    frames = np.arange(int(5 * FPS))
    positions = np.array([30.0, 30.0]) + rng.normal(0, 0.1, size=(len(frames), 2))

    metrics = compute_movement_metrics(make_player(frames, positions), FPS)

    # A standing player: raw frame-to-frame speed would be ~13 km/h of noise.
    assert metrics.max_speed_kmh < 5.0


def test_tracking_glitch_is_ignored():
    frames, positions = straight_run(speed_mps=3.0, seconds=4)
    positions[60:] += [25.0, 0.0]      # ID switch: 25 m teleport in one frame

    metrics = compute_movement_metrics(make_player(frames, positions), FPS, smoothing_s=0)

    assert metrics.max_speed_kmh == pytest.approx(3.0 * MPS_TO_KMH, rel=0.02)
    assert metrics.distance_m < 15


def test_off_pitch_positions_are_excluded():
    frames, positions = straight_run(speed_mps=4.0, seconds=2)
    player = make_player(frames, positions)
    player.pitch_positions = [(np.nan, np.nan)] * len(frames)

    metrics = compute_movement_metrics(player, FPS)

    assert metrics.distance_m == 0
    assert metrics.time_on_pitch_s == 0


def test_segments_fill_small_gaps_and_split_large_ones():
    frames = [0, 1, 3, 4, 20, 21, 22]
    positions = [(float(f), 0.0) for f in frames]

    segments = pitch_segments(make_player(frames, positions), max_gap_frames=5)

    assert len(segments) == 2
    np.testing.assert_array_equal(segments[0][0], [0, 1, 2, 3, 4])
    assert segments[0][1][2] == pytest.approx((2.0, 0.0))


def test_smoothing_keeps_length_and_straight_lines():
    xy = np.column_stack([np.arange(20.0), np.zeros(20)])
    smooth = smooth_positions(xy, window=5)

    assert smooth.shape == xy.shape
    np.testing.assert_allclose(smooth[2:-2], xy[2:-2])


def test_pixel_distance_still_available():
    player = make_player([0, 1], [(0.0, 0.0), (3.0, 4.0)])
    assert calculate_total_distance(player) == pytest.approx(5.0)
