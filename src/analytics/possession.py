"""Ball possession, passes and turnovers from player and ball tracks.

Possession is decided in the image, not on the pitch: the ball belongs to the
player whose feet are closest to it, if that distance is small compared to
the player's height in pixels. This is independent of the pitch calibration
and tolerant to a ball that is slightly off the ground.

Raw per-frame ownership flickers, so it is cleaned in two steps:
    1. short gaps where nobody owns the ball are bridged when the same
       player has it before and after (a dribble);
    2. ownership spells shorter than min_spell_frames are discarded (the ball
       just passing someone's feet).

Events are then read off the cleaned sequence:
    same team, different player  -> pass
    different team               -> turnover
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Spell:
    player_id: int
    team: str
    start: int      # first frame
    end: int        # last frame (inclusive)


@dataclass(frozen=True)
class Event:
    kind: str       # "pass" or "turnover"
    frame: int      # frame at which the receiver gets the ball
    from_player: int
    to_player: int
    from_team: str
    to_team: str


def owner_per_frame(players_by_frame, ball_by_frame, max_distance_ratio=0.7):
    """Return {frame: player_id} for frames where someone has the ball.

    players_by_frame: {frame: [(player_id, (x1, y1, x2, y2)), ...]}  (officials excluded)
    ball_by_frame:    {frame: (x, y)}
    max_distance_ratio: largest ball-to-feet distance, as a fraction of the player's box height.
    """

    owners = {}

    for frame, (ball_x, ball_y) in ball_by_frame.items():
        best_id, best_ratio = None, max_distance_ratio

        for player_id, (x1, y1, x2, y2) in players_by_frame.get(frame, []):
            height = max(y2 - y1, 1.0)
            feet_x, feet_y = (x1 + x2) / 2, y2
            ratio = float(np.hypot(ball_x - feet_x, ball_y - feet_y)) / height

            if ratio <= best_ratio:
                best_id, best_ratio = player_id, ratio

        if best_id is not None:
            owners[frame] = best_id

    return owners


def possession_spells(owners, team_of, max_bridge_frames=15, min_spell_frames=4):
    """Clean per-frame ownership into a list of Spell, in time order."""

    if not owners:
        return []

    # Runs of consecutive frames with the same owner.
    runs = []
    for frame in sorted(owners):
        player = owners[frame]
        if runs and runs[-1][0] == player and frame - runs[-1][2] == 1:
            runs[-1][2] = frame
        else:
            runs.append([player, frame, frame])

    # 1. Bridge short gaps between runs of the same player.
    bridged = [runs[0]]
    for player, start, end in runs[1:]:
        last = bridged[-1]
        if last[0] == player and start - last[2] - 1 <= max_bridge_frames:
            last[2] = end
        else:
            bridged.append([player, start, end])

    # 2. Drop spells that are too short to be real control of the ball.
    kept = [run for run in bridged if run[2] - run[1] + 1 >= min_spell_frames]

    # Removing a short spell can leave two spells of the same player side by side.
    merged = []
    for player, start, end in kept:
        if merged and merged[-1][0] == player and start - merged[-1][2] - 1 <= max_bridge_frames:
            merged[-1][2] = end
        else:
            merged.append([player, start, end])

    return [Spell(player, team_of.get(player, ""), start, end) for player, start, end in merged]


def detect_events(spells, max_gap_frames=90):
    """Passes and turnovers between consecutive spells (ball travel time up to max_gap_frames)."""

    events = []

    for previous, current in zip(spells[:-1], spells[1:]):
        if current.start - previous.end > max_gap_frames or previous.player_id == current.player_id:
            continue

        kind = "pass" if previous.team == current.team and previous.team else "turnover"
        events.append(Event(
            kind, current.start, previous.player_id, current.player_id, previous.team, current.team
        ))

    return events


def team_possession(spells):
    """Share of possession time per team (fractions summing to 1)."""

    frames = Counter()
    for spell in spells:
        if spell.team:
            frames[spell.team] += spell.end - spell.start + 1

    total = sum(frames.values())

    return {team: count / total for team, count in sorted(frames.items())} if total else {}


def passing_network(events):
    """{(from_player, to_player): number of passes}."""

    return dict(Counter((e.from_player, e.to_player) for e in events if e.kind == "pass"))
