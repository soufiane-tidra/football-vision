import pytest

from src.analytics.possession import (
    detect_events,
    owner_per_frame,
    passing_network,
    possession_spells,
    team_possession,
)


TEAM = {1: "team_1", 2: "team_1", 3: "team_2"}


def box_at(x, y_feet, height=80.0, width=30.0):
    return (x - width / 2, y_feet - height, x + width / 2, y_feet)


def test_owner_is_the_closest_player_within_reach():
    players = {0: [(1, box_at(100, 500)), (2, box_at(400, 500))]}

    assert owner_per_frame(players, {0: (110.0, 505.0)}) == {0: 1}
    assert owner_per_frame(players, {0: (390.0, 495.0)}) == {0: 2}
    assert owner_per_frame(players, {0: (250.0, 500.0)}) == {}          # between them: nobody


def test_reach_scales_with_player_size():
    near_camera = {0: [(1, box_at(100, 500, height=160))]}
    far_away = {0: [(1, box_at(100, 500, height=40))]}
    ball = {0: (160.0, 500.0)}                                           # 60 px from the feet

    assert owner_per_frame(near_camera, ball) == {0: 1}
    assert owner_per_frame(far_away, ball) == {}


def owners_from(sequence):
    """sequence: list of (player_id or None, n_frames)."""
    owners, frame = {}, 0
    for player, n in sequence:
        for _ in range(n):
            if player is not None:
                owners[frame] = player
            frame += 1
    return owners


def test_dribble_gaps_are_bridged_and_touches_ignored():
    owners = owners_from([(1, 20), (None, 5), (1, 20), (3, 2), (None, 10), (2, 30)])
    spells = possession_spells(owners, TEAM)

    assert [(s.player_id, s.start, s.end) for s in spells] == [(1, 0, 44), (2, 57, 86)]


def test_pass_and_turnover_detection():
    owners = owners_from([(1, 30), (None, 20), (2, 30), (None, 15), (3, 30), (None, 10), (3, 10)])
    spells = possession_spells(owners, TEAM)
    events = detect_events(spells)

    assert [(e.kind, e.from_player, e.to_player) for e in events] == [("pass", 1, 2), ("turnover", 2, 3)]
    assert events[0].frame == 50
    assert passing_network(events) == {(1, 2): 1}


def test_long_silence_is_not_a_pass():
    owners = owners_from([(1, 30), (None, 200), (2, 30)])
    assert detect_events(possession_spells(owners, TEAM)) == []


def test_team_possession_shares():
    owners = owners_from([(1, 30), (None, 10), (2, 30), (None, 10), (3, 20)])
    shares = team_possession(possession_spells(owners, TEAM))

    assert shares == pytest.approx({"team_1": 0.75, "team_2": 0.25})
    assert team_possession([]) == {}
