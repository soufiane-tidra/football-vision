import cv2
import numpy as np

from src.classification.identity import (
    GOALKEEPER,
    PLAYER,
    REFEREE,
    assign_identities,
    in_goal_area,
    is_pitch_side_staff,
)
from src.classification.team_classifier import OTHER, TeamClassifier
from src.tracking.player import PlayerTrack
from src.tracking.stitching import find_links, stitch_tracks


FPS = 30.0


def lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


RED, NAVY, CYAN, GREEN = lab((40, 40, 210)), lab((90, 50, 40)), lab((220, 200, 60)), lab((120, 230, 150))


def make_track(track_id, frames, start, velocity=(0.0, 0.0), role="player", confidence=0.85):
    track = PlayerTrack(track_id=track_id)
    for i, frame in enumerate(frames):
        x = start[0] + velocity[0] * i / FPS
        y = start[1] + velocity[1] * i / FPS
        track.add_detection(frame, 0.0, 0.0, confidence, role, box=(0.0, 0.0, 10.0, 20.0))
        track.pitch_positions.append((x, y))
    return track


# ---------- team classifier ----------

def test_three_kits_give_two_teams_and_others():
    rng = np.random.default_rng(0)
    colors = {}
    for i in range(10):
        colors[i] = RED + rng.normal(0, 2, 3)
        colors[100 + i] = NAVY + rng.normal(0, 2, 3)
    for i in range(4):
        colors[200 + i] = CYAN + rng.normal(0, 2, 3)

    teams = TeamClassifier().fit_predict(colors)

    assert len({teams[i] for i in range(10)}) == 1 and teams[0] != OTHER
    assert len({teams[100 + i] for i in range(10)}) == 1 and teams[100] not in (OTHER, teams[0])
    assert all(teams[200 + i] == OTHER for i in range(4))


def test_two_kits_only_are_not_split_into_three():
    rng = np.random.default_rng(1)
    colors = {i: RED + rng.normal(0, 3, 3) for i in range(12)}
    colors.update({100 + i: NAVY + rng.normal(0, 3, 3) for i in range(12)})

    teams = TeamClassifier().fit_predict(colors)

    assert sum(team == OTHER for team in teams.values()) <= 1
    assert len({teams[i] for i in range(12)} - {OTHER}) == 1


def test_brightness_change_keeps_the_team():
    rng = np.random.default_rng(2)
    colors = {i: RED + rng.normal(0, 2, 3) for i in range(10)}
    colors.update({100 + i: NAVY + rng.normal(0, 2, 3) for i in range(10)})
    colors.update({200 + i: CYAN + rng.normal(0, 2, 3) for i in range(3)})

    classifier = TeamClassifier()
    teams = classifier.fit_predict(colors)

    in_shadow = NAVY.copy()
    in_shadow[0] += 20                      # same hue, brighter (fog / floodlight)
    assert classifier.predict(in_shadow) == teams[100]


# ---------- identities ----------

def build_scene():
    tracks, colors = {}, {}
    frames = list(range(90))

    for i in range(8):                                   # team defending the left goal
        tracks[i] = make_track(i, frames, (25 + 2 * i, 20 + 3 * i))
        colors[i] = NAVY
    for i in range(8):
        tracks[100 + i] = make_track(100 + i, frames, (60 + 2 * i, 20 + 3 * i))
        colors[100 + i] = RED

    tracks[200] = make_track(200, frames, (50, 34))                      # main referee, detector says "player"
    colors[200] = CYAN
    tracks[201] = make_track(201, frames, (30, 68.5), role="referee")    # linesman, detector is right
    colors[201] = CYAN
    tracks[300] = make_track(300, frames, (4, 34))                       # goalkeeper in a different kit
    colors[300] = GREEN
    tracks[301] = make_track(301, frames, (5, 36))                       # goalkeeper fragment that looks navy
    colors[301] = NAVY

    return tracks, colors


def test_identities_combine_color_role_and_position():
    tracks, colors = build_scene()
    identities, _ = assign_identities(tracks, colors)

    left_team = identities[0].team
    assert all(identities[i] == identities[0] for i in range(8))
    assert identities[0].role == PLAYER
    assert identities[100].team == 1 - left_team

    assert identities[200].role == REFEREE and identities[200].team is None    # by color
    assert identities[201].role == REFEREE                                     # by detector
    assert identities[300].role == GOALKEEPER and identities[300].team == left_team
    assert identities[301].role == GOALKEEPER                                  # by position, despite the color


def test_goal_area_detection():
    assert in_goal_area(np.array([3.0, 34.0])) == "left"
    assert in_goal_area(np.array([102.0, 30.0])) == "right"
    assert in_goal_area(np.array([3.0, 5.0])) is None        # near the corner flag
    assert in_goal_area(np.array([50.0, 34.0])) is None
    assert in_goal_area(None) is None


def test_pitch_side_staff_rules():
    frames = list(range(150))

    coach = make_track(1, frames, (46.0, 68.0))                              # still, on the touchline
    winger = make_track(2, frames, (40.0, 67.0), velocity=(5.0, -0.5))       # on the wing but running
    midfielder = make_track(3, frames, (50.0, 30.0))                         # standing, but in the middle
    cameraman = make_track(4, frames, (30.0, 60.0), confidence=0.45)         # low detector confidence

    assert is_pitch_side_staff(coach)
    assert not is_pitch_side_staff(winger)
    assert not is_pitch_side_staff(midfielder)
    assert is_pitch_side_staff(cameraman)


# ---------- stitching ----------

def test_fragments_of_one_player_are_merged():
    a = make_track(1, range(0, 60), (30.0, 30.0), velocity=(3.0, 0.0))
    b = make_track(2, range(75, 150), (37.5, 30.0), velocity=(3.0, 0.0))     # reappears 0.5 s later, on course
    groups = {1: "team_a", 2: "team_a"}

    merged, mapping = stitch_tracks({1: a, 2: b}, groups, FPS)

    assert list(merged) == [1] and mapping == {1: 1, 2: 1}
    assert merged[1].detection_count == 135
    assert merged[1].frames == sorted(merged[1].frames)


def test_no_merge_across_teams_or_impossible_distances():
    a = make_track(1, range(0, 60), (30.0, 30.0))
    other_team = make_track(2, range(70, 120), (30.0, 30.0))
    too_far = make_track(3, range(70, 120), (70.0, 30.0))                    # 40 m in 0.4 s
    too_late = make_track(4, range(400, 450), (30.0, 30.0))                  # 11 s later
    groups = {1: "a", 2: "b", 3: "a", 4: "a"}

    assert find_links({1: a, 2: other_team, 3: too_far, 4: too_late}, groups, FPS) == []


def test_closest_candidate_wins_and_chains_are_followed():
    a = make_track(1, range(0, 30), (30.0, 30.0))
    near = make_track(2, range(40, 70), (31.0, 30.0))
    far = make_track(3, range(40, 70), (36.0, 30.0))
    later = make_track(4, range(80, 110), (31.5, 30.0))
    groups = dict.fromkeys((1, 2, 3, 4), "a")

    merged, mapping = stitch_tracks({1: a, 2: near, 3: far, 4: later}, groups, FPS)

    assert mapping[2] == 1 and mapping[4] == 1          # 1 -> 2 -> 4
    assert mapping[3] == 3                              # left alone
    assert merged[1].detection_count == 90


def test_overlapping_fragments_are_merged_without_duplicate_frames():
    a = make_track(1, range(0, 100), (30.0, 30.0))
    b = make_track(2, range(94, 200), (30.5, 30.0))      # tracker held two IDs for 6 frames
    groups = {1: "a", 2: "a"}

    merged, _ = stitch_tracks({1: a, 2: b}, groups, FPS)

    assert list(merged) == [1]
    assert merged[1].frames == list(range(200))
    assert len(merged[1].pitch_positions) == len(merged[1].boxes) == 200


def test_parallel_tracks_are_not_merged():
    a = make_track(1, range(0, 100), (30.0, 30.0))
    b = make_track(2, range(90, 100), (30.5, 30.0))      # ends with a: a duplicate, not a continuation

    assert find_links({1: a, 2: b}, {1: "a", 2: "a"}, FPS) == []
