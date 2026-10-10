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


# ---------- track loading / roles ----------

def write_tracks(path, rows):
    header = "frame,track_id,class_id,class_name,confidence,x1,y1,x2,y2,center_x,center_y\n"
    lines = [f"{f},{t},0,{c},0.9,0,0,10,20,5,10\n" for f, t, c in rows]
    path.write_text(header + "".join(lines))


def test_role_is_majority_class_and_referees_are_excluded(tmp_path):
    from src.tracking.loader import load_player_tracks

    path = tmp_path / "tracks.csv"
    write_tracks(path, (
        [(f, 1, "player") for f in range(8)] + [(8, 1, "referee")]          # player, one bad frame
        + [(f, 2, "referee") for f in range(8)] + [(8, 2, "player")]        # referee, one bad frame
        + [(f, 3, "goalkeeper") for f in range(5)]
        + [(f, 4, "ball") for f in range(5)]
    ))

    players = load_player_tracks(path)
    assert set(players) == {1, 3}
    assert players[1].role == "player" and players[3].role == "goalkeeper"
    assert players[1].positions[0] == (5.0, 20.0)                            # feet = bottom-center

    everyone = load_player_tracks(path, roles=None)
    assert set(everyone) == {1, 2, 3}                                        # the ball is never a person
    assert everyone[2].role == "referee"


def test_generic_person_class_still_works(tmp_path):
    from src.tracking.loader import load_player_tracks

    path = tmp_path / "tracks.csv"
    write_tracks(path, [(0, 7, "person"), (1, 7, "person"), (5, 7, "person")])

    assert set(load_player_tracks(path)) == {7}
    assert load_player_tracks(path, max_frames=2)[7].frames == [0, 1]


def test_top_speed_must_be_sustained():
    frames, positions = straight_run(speed_mps=4.0, seconds=4)
    positions[60:] += [0.2, 0.0]       # one noisy step: +0.2 m in a single frame = 6 m/s extra

    spiky = make_player(frames, positions)

    instant = compute_movement_metrics(spiky, FPS, smoothing_s=0, sustain_s=0)
    sustained = compute_movement_metrics(spiky, FPS, smoothing_s=0, sustain_s=0.5)

    assert instant.max_speed_mps > 9
    assert sustained.max_speed_mps == pytest.approx(4.0, abs=0.7)


def test_sprint_and_high_speed_distance():
    # 3 s jog at 3 m/s, then a 2 s sprint at 8 m/s (28.8 km/h), then 3 s jog.
    speeds = [3.0] * 90 + [8.0] * 60 + [3.0] * 90
    xs = np.concatenate([[0.0], np.cumsum(np.array(speeds) / FPS)])
    positions = np.column_stack([10 + xs, np.full(len(xs), 30.0)])

    metrics = compute_movement_metrics(make_player(np.arange(len(xs)), positions), FPS, smoothing_s=0)

    assert metrics.sprints == 1
    assert metrics.sprint_distance_m == pytest.approx(16.0, abs=2.5)       # 8 m/s for ~2 s
    assert metrics.high_speed_distance_m >= metrics.sprint_distance_m
    assert metrics.max_speed_kmh == pytest.approx(28.8, abs=0.5)


def test_jogging_player_has_no_sprints():
    frames, positions = straight_run(speed_mps=3.5, seconds=6)
    metrics = compute_movement_metrics(make_player(frames, positions), FPS)

    assert metrics.sprints == 0
    assert metrics.sprint_distance_m == 0
    assert metrics.high_speed_distance_m == 0


def test_motion_profile_gives_position_and_speed_per_frame():
    from src.analytics.movement import motion_profile

    frames, positions = straight_run(speed_mps=5.0, seconds=3)
    xy, kmh = motion_profile(make_player(frames, positions), FPS)

    assert set(xy) == set(kmh) == set(int(f) for f in frames)
    assert kmh[45] == pytest.approx(18.0, abs=0.5)
    assert xy[45] == pytest.approx((10.0 + 5.0 * 45 / FPS, 30.0), abs=0.05)
