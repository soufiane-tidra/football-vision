"""Merge fragmented tracks that belong to the same person.

The tracker gives a new ID whenever it loses someone (occlusion, missed
detections). Two fragments are joined when:
    - they have the same identity group (team + role),
    - the second starts after the first ends, within max_gap_s (a short
      overlap is allowed: the tracker sometimes holds two IDs on one player),
    - the distance between the end of the first and the start of the second
      is physically reachable:  distance <= slack_m + max_speed_mps * gap.

Candidate links are taken greedily from the cheapest (closest) one; every
fragment gets at most one predecessor and one successor.
"""

import numpy as np

from src.tracking.player import PlayerTrack


def _edge_position(track, at_end, samples=5):
    """Robust pitch position at the start or end of a track (median of a few valid points)."""

    points = np.asarray(track.pitch_positions, dtype=float).reshape(-1, 2)
    valid = points[~np.isnan(points).any(axis=1)]

    if len(valid) == 0:
        return None

    return np.median(valid[-samples:] if at_end else valid[:samples], axis=0)


def find_links(tracks, group_of, fps, max_gap_s=3.0, max_speed_mps=8.0, slack_m=3.0, max_overlap_s=0.5):
    """Return a list of (earlier_id, later_id) links between fragments."""

    ends = {t: _edge_position(track, at_end=True) for t, track in tracks.items()}
    starts = {t: _edge_position(track, at_end=False) for t, track in tracks.items()}

    candidates = []

    for a, track_a in tracks.items():
        if ends[a] is None:
            continue

        for b, track_b in tracks.items():
            if a == b or starts[b] is None or group_of[a] != group_of[b]:
                continue

            gap_s = (track_b.first_frame - track_a.last_frame) / fps
            if not -max_overlap_s <= gap_s <= max_gap_s:
                continue

            # b must really come later, not be a track that runs alongside a.
            if track_b.first_frame <= track_a.first_frame or track_b.last_frame <= track_a.last_frame:
                continue

            distance = float(np.linalg.norm(starts[b] - ends[a]))
            if distance <= slack_m + max_speed_mps * max(gap_s, 0.0):
                # Prefer close fragments, then short gaps.
                candidates.append((distance + abs(gap_s), a, b))

    links = []
    has_successor, has_predecessor = set(), set()

    for _, a, b in sorted(candidates):
        if a in has_successor or b in has_predecessor:
            continue
        links.append((a, b))
        has_successor.add(a)
        has_predecessor.add(b)

    return links


def stitch_tracks(tracks, group_of, fps, **kwargs):
    """Merge linked fragments. Returns (merged tracks keyed by new id, old id -> new id).

    The new id of a chain is the id of its first fragment.
    """

    successor = dict(find_links(tracks, group_of, fps, **kwargs))
    is_successor = set(successor.values())

    merged = {}
    mapping = {}

    for head in tracks:
        if head in is_successor:
            continue

        chain = [head]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])

        merged[head] = _merge([tracks[t] for t in chain], head)
        for track_id in chain:
            mapping[track_id] = head

    return merged, mapping


def _merge(fragments, new_id):
    result = PlayerTrack(track_id=new_id)

    for fragment in fragments:
        # Where fragments overlap in time, keep the earlier fragment's detections.
        last_frame = result.frames[-1] if result.frames else -1
        keep = [i for i, frame in enumerate(fragment.frames) if frame > last_frame]

        for name in ("frames", "positions", "confidences", "boxes", "pitch_positions"):
            values = getattr(fragment, name)
            if values:
                getattr(result, name).extend(values[i] for i in keep)

        result.class_counts.update(fragment.class_counts)

    return result
