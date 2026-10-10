import numpy as np

from src.ball.trajectory import interpolate_gaps, select_trajectory


def ball_path(n_frames, start=(200.0, 500.0), velocity=(12.0, -3.0)):
    return {f: (start[0] + velocity[0] * f, start[1] + velocity[1] * f) for f in range(n_frames)}


def test_real_ball_is_chosen_over_random_false_positives():
    rng = np.random.default_rng(0)
    truth = ball_path(120)

    candidates = {}
    for frame, (x, y) in truth.items():
        found = [(x + rng.normal(0, 1), y + rng.normal(0, 1), 0.3)]
        for _ in range(4):                                             # more confident, but jumping around
            found.append((rng.uniform(0, 1900), rng.uniform(0, 1000), 0.6))
        candidates[frame] = found

    points = select_trajectory(candidates)

    assert len(points) >= 110
    errors = [np.hypot(p.x - truth[p.frame][0], p.y - truth[p.frame][1]) for p in points]
    assert np.mean(np.array(errors) < 5) > 0.95


def test_missed_detections_are_interpolated():
    truth = ball_path(60)
    candidates = {f: [(x, y, 0.5)] for f, (x, y) in truth.items() if not 20 <= f < 30}   # 10 frames missing

    points = interpolate_gaps(select_trajectory(candidates))

    assert [p.frame for p in points] == list(range(60))
    filled = [p for p in points if p.interpolated]
    assert [p.frame for p in filled] == list(range(20, 30))
    assert all(np.hypot(p.x - truth[p.frame][0], p.y - truth[p.frame][1]) < 1e-6 for p in filled)


def test_impossible_jump_starts_a_new_segment_without_interpolation():
    first = {f: [(100.0 + 5 * f, 300.0, 0.6)] for f in range(40)}
    second = {f: [(1500.0 + 5 * f, 800.0, 0.6)] for f in range(42, 80)}        # 1200 px away 2 frames later

    points = interpolate_gaps(select_trajectory({**first, **second}))

    assert len({p.segment for p in points}) == 2
    assert not any(p.interpolated for p in points)                             # no line drawn across the jump
    assert not any(p.frame in (40, 41) for p in points)


def test_isolated_blips_are_dropped():
    truth = ball_path(50)
    candidates = {f: [(x, y, 0.4)] for f, (x, y) in truth.items()}
    candidates[300] = [(900.0, 900.0, 0.9)]                                    # one lonely detection much later

    points = select_trajectory(candidates)

    assert max(p.frame for p in points) < 50


def test_no_candidates():
    assert select_trajectory({0: [], 1: []}) == []
    assert interpolate_gaps([]) == []


def test_weak_stretch_is_dropped_but_confident_one_is_kept():
    boot = {f: [(300.0 + 2 * f, 400.0, 0.06)] for f in range(60)}              # continuous but weak
    ball = {f: [(900.0 + 8 * (f - 100), 500.0, 0.45)] for f in range(100, 180)}

    points = select_trajectory({**boot, **ball}, min_segment_confidence=0.2)

    assert points and min(p.frame for p in points) == 100


def test_candidates_off_the_pitch_and_on_markings_are_removed():
    from src.ball.trajectory import filter_candidates

    identity = np.array([np.eye(3)] * 200)                 # pixels == meters for this test
    candidates = {}
    for frame in range(200):
        candidates[frame] = [
            (30.0 + 0.2 * frame, 40.0, 0.4),               # moving ball
            (50.0, 75.0, 0.4),                             # outside the pitch (y > 68)
            (11.0, 34.0, 0.15),                            # penalty spot, weak, always there
        ]
    candidates[10].append((80.0, 20.0, 0.9))               # confident ball lying on a static cell is kept

    kept = filter_candidates(candidates, identity, static_frames=120)

    assert all(len(found) == 1 for frame, found in kept.items() if frame != 10)
    assert all(found[0][1] == 40.0 for frame, found in kept.items() if frame != 10)
    assert len(kept[10]) == 2


def test_static_stretch_is_dropped_even_if_confident():
    from src.ball.trajectory import drop_static_segments

    identity = np.array([np.eye(3)] * 300)                                     # pixels == meters
    rng = np.random.default_rng(0)

    marking = {f: [(46.0 + rng.normal(0, 0.5), 67.0 + rng.normal(0, 0.5), 0.6)] for f in range(100)}
    ball = {f: [(20.0 + 0.3 * (f - 150), 30.0, 0.4)] for f in range(150, 250)}  # travels 30 m

    points = select_trajectory({**marking, **ball}, max_speed_px=3.0, slack_px=1.0)
    moving = drop_static_segments(points, identity, min_travel_m=5.0)

    assert {p.segment for p in points} != {p.segment for p in moving}
    assert moving and min(p.frame for p in moving) == 150
