"""Decide each track's team and role by combining three signals.

    jersey color     two team clusters; anything far from both is not an outfield player
    detector role    majority class of the track (player / goalkeeper / referee)
    pitch position   a color outlier that lives near a goal is a goalkeeper
"""

from dataclasses import dataclass

import numpy as np

from src.classification.team_classifier import OTHER, TeamClassifier
from src.pitch.landmarks import PitchDimensions


PLAYER = "player"
GOALKEEPER = "goalkeeper"
REFEREE = "referee"


@dataclass(frozen=True)
class Identity:
    role: str
    team: int | None      # 0 or 1; None for referees


def median_pitch_position(track):
    points = np.asarray(track.pitch_positions, dtype=float).reshape(-1, 2)
    points = points[~np.isnan(points).any(axis=1)]

    return np.median(points, axis=0) if len(points) else None


def is_pitch_side_staff(track, dims: PitchDimensions = PitchDimensions(),
                        band_m=1.5, min_share=0.8, max_spread_m=3.0, min_confidence=0.6):
    """Coaches, substitutes and cameramen: not players, even if their clothes match a team.

    They either are detected with low confidence (the detector was trained on
    players), or stand still right at / beyond a touchline for the whole track.
    """

    if track.confidences and float(np.mean(track.confidences)) < min_confidence:
        return True

    points = np.asarray(track.pitch_positions, dtype=float).reshape(-1, 2)
    points = points[~np.isnan(points).any(axis=1)]

    if len(points) == 0:
        return True

    at_touchline = (points[:, 1] <= band_m) | (points[:, 1] >= dims.width - band_m)
    stationary = float(np.linalg.norm(points.std(axis=0))) <= max_spread_m

    return bool(at_touchline.mean() >= min_share and stationary)


def in_goal_area(position, dims: PitchDimensions = PitchDimensions(), margin_m=3.0):
    """Which goal area a position is in: "left", "right" or None."""

    if position is None:
        return None

    depth = dims.goal_area_depth + margin_m
    half_width = dims.goal_area_width / 2 + margin_m

    if abs(position[1] - dims.width / 2) > half_width:
        return None
    if position[0] <= depth:
        return "left"
    if position[0] >= dims.length - depth:
        return "right"
    return None


def _team_defending_left(tracks, team_of, dims):
    """The team whose deepest players are closest to the left goal.

    Because of the offside rule the last outfield players in front of a goal
    belong to the team defending it, so we compare how deep each team gets
    (5th percentile of x) rather than where it stands on average.
    """

    depth = {}

    for team in (0, 1):
        xs = np.concatenate([
            np.asarray(track.pitch_positions, dtype=float).reshape(-1, 2)[:, 0]
            for t, track in tracks.items() if team_of[t] == team
        ] or [np.array([])])
        xs = xs[~np.isnan(xs)]
        depth[team] = float(np.percentile(xs, 5)) if len(xs) else dims.length / 2

    return min(depth, key=depth.get)


def assign_identities(
    tracks,
    colors,
    dims: PitchDimensions = PitchDimensions(),
    min_fit_detections=30,
    goal_zone_m=20.0,
):
    """tracks: id -> PlayerTrack (with pitch_positions); colors: id -> Lab color.

    Returns (id -> Identity, fitted TeamClassifier). Tracks without a color
    sample are skipped.
    """

    usable = {t: tracks[t] for t in colors if t in tracks}

    # Team colors are learned from long tracks the detector calls "player".
    fit_ids = [
        t for t, track in usable.items()
        if track.role in (PLAYER, "person") and track.detection_count >= min_fit_detections
    ]
    if len(fit_ids) < 4:
        fit_ids = list(usable)

    classifier = TeamClassifier()
    classifier.fit_predict({t: colors[t] for t in fit_ids})

    team_of = {t: classifier.predict(colors[t]) for t in usable}

    left_team = _team_defending_left(usable, team_of, dims)

    identities = {}

    for track_id, track in usable.items():
        team = team_of[track_id]

        if track.role == REFEREE:
            identities[track_id] = Identity(REFEREE, None)
            continue

        # Whoever lives in a goal area is the goalkeeper, whatever the kit color.
        goal_side = in_goal_area(median_pitch_position(track), dims)
        if goal_side is not None:
            keeper_team = left_team if goal_side == "left" else 1 - left_team
            identities[track_id] = Identity(GOALKEEPER, keeper_team)
            continue

        if team != OTHER:
            role = GOALKEEPER if track.role == GOALKEEPER else PLAYER
            identities[track_id] = Identity(role, team)
            continue

        # Kit matches neither team: goalkeeper or match official.
        position = median_pitch_position(track)
        near_left = position is not None and position[0] <= goal_zone_m
        near_right = position is not None and position[0] >= dims.length - goal_zone_m

        if near_left or near_right:
            keeper_team = left_team if near_left else 1 - left_team
            identities[track_id] = Identity(GOALKEEPER, keeper_team)
        elif track.role == GOALKEEPER and position is not None:
            keeper_team = left_team if position[0] < dims.length / 2 else 1 - left_team
            identities[track_id] = Identity(GOALKEEPER, keeper_team)
        else:
            identities[track_id] = Identity(REFEREE, None)

    return identities, classifier
