"""Pick the real ball out of noisy per-frame candidates.

Every frame has zero or more ball candidates (the real ball plus false
positives such as socks, heads or line markings). The real ball is the one
that forms a long, physically continuous trajectory. Dynamic programming
finds the best-scoring path through the candidates:

    score(path) = sum of detection confidences
                  + a bonus for every link between consecutive detections
                  - a penalty per skipped frame inside a link
                  - a cost growing with the distance jumped (smooth paths win)
                  - a penalty every time the path has to "restart" somewhere
                    that is not reachable from the previous detection

A link between two candidates is allowed only if the ball could have covered
the distance in the elapsed frames.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BallPoint:
    frame: int
    x: float
    y: float
    confidence: float      # 0.0 for interpolated points
    interpolated: bool
    segment: int           # points of one continuous stretch share a segment id


def select_trajectory(
    candidates,
    max_speed_px=70.0,
    slack_px=15.0,
    max_link_gap=30,
    link_bonus=0.5,
    gap_penalty=0.05,
    distance_penalty=0.5,
    restart_penalty=3.0,
    min_segment_points=3,
    min_segment_confidence=0.0,
):
    """candidates: {frame: [(x, y, confidence, ...), ...]}. Returns a list of BallPoint (no interpolation).

    max_speed_px: largest ball displacement per frame in pixels (ball speed plus camera pan).
    min_segment_confidence: stretches whose mean confidence is below this are dropped; a
        continuous but weak stretch is usually a boot or a sock, and "no ball" is the honest answer.
    """

    nodes = sorted(
        (int(frame), float(c[0]), float(c[1]), float(c[2]))
        for frame, found in candidates.items() for c in found
    )

    if not nodes:
        return []

    frames = np.array([n[0] for n in nodes])
    xy = np.array([[n[1], n[2]] for n in nodes])
    conf = np.array([n[3] for n in nodes])
    n = len(nodes)

    score = np.zeros(n)
    previous = np.full(n, -1)
    linked = np.zeros(n, dtype=bool)

    # Best path score among nodes of strictly earlier frames, for restarts.
    best_before_score, best_before_node = -np.inf, -1
    pending_score, pending_node, pending_frame = -np.inf, -1, frames[0]

    first_of_window = 0

    for j in range(n):
        if frames[j] != pending_frame:
            if pending_score > best_before_score:
                best_before_score, best_before_node = pending_score, pending_node
            pending_score, pending_node, pending_frame = -np.inf, -1, frames[j]

        best, best_prev, is_link = 0.0, -1, False      # start a new path here

        if best_before_node >= 0 and best_before_score - restart_penalty > best:
            best, best_prev = best_before_score - restart_penalty, best_before_node

        while frames[j] - frames[first_of_window] > max_link_gap:
            first_of_window += 1

        window = np.arange(first_of_window, j)
        window = window[frames[window] < frames[j]]

        if len(window):
            dt = frames[j] - frames[window]
            distance = np.linalg.norm(xy[window] - xy[j], axis=1)
            reach = slack_px + max_speed_px * dt
            reachable = distance <= reach
            if reachable.any():
                options = score[window] + link_bonus - gap_penalty * (dt - 1) - distance_penalty * distance / reach
                options[~reachable] = -np.inf
                k = int(options.argmax())
                if options[k] > best:
                    best, best_prev, is_link = float(options[k]), int(window[k]), True

        score[j] = conf[j] + best
        previous[j] = best_prev
        linked[j] = is_link

        if score[j] > pending_score:
            pending_score, pending_node = score[j], j

    # Backtrack from the best end point.
    path = []
    j = int(score.argmax())
    while j >= 0:
        path.append(j)
        j = int(previous[j])
    path.reverse()

    points = []
    segment = 0
    for position, j in enumerate(path):
        if position > 0 and not linked[j]:
            segment += 1
        points.append(BallPoint(int(frames[j]), float(xy[j, 0]), float(xy[j, 1]), float(conf[j]), False, segment))

    # Very short or very weak stretches are more likely noise than ball.
    segments = np.array([p.segment for p in points])
    confidences = np.array([p.confidence for p in points])
    sizes = np.bincount(segments)
    means = np.bincount(segments, weights=confidences) / np.maximum(sizes, 1)

    return [
        p for p in points
        if sizes[p.segment] >= min_segment_points and means[p.segment] >= min_segment_confidence
    ]


def interpolate_gaps(points, max_gap=30):
    """Fill missing frames inside each segment by linear interpolation."""

    result = []

    for previous, current in zip([None] + points[:-1], points):
        if previous is not None and previous.segment == current.segment:
            gap = current.frame - previous.frame
            if 1 < gap <= max_gap:
                for frame in range(previous.frame + 1, current.frame):
                    t = (frame - previous.frame) / gap
                    result.append(BallPoint(
                        frame,
                        previous.x + t * (current.x - previous.x),
                        previous.y + t * (current.y - previous.y),
                        0.0, True, current.segment,
                    ))
        result.append(current)

    return result


def filter_candidates(
    candidates,
    homographies,
    pitch_length=105.0,
    pitch_width=68.0,
    margin_m=1.5,
    cell_m=2.0,
    static_frames=120,
    keep_confidence=0.5,
):
    """Remove candidates that cannot be the ball in play, using pitch coordinates.

    - outside the pitch: technical-area markings, objects behind the lines;
    - painted markings: a spot of the pitch that "contains a ball" in a very
      large number of frames (static_frames) is a marking, not a ball. A real
      ball lying still is kept as long as it is detected confidently.

    candidates: {frame: [(x, y, confidence, ...), ...]}; homographies: (N, 3, 3) image -> pitch.
    """

    projected = {}

    for frame, found in candidates.items():
        if not found or frame >= len(homographies):
            projected[frame] = []
            continue

        pixels = np.array([[c[0], c[1]] for c in found], dtype=np.float64).reshape(-1, 1, 2)
        pitch = cv2.perspectiveTransform(pixels, np.asarray(homographies[frame], dtype=np.float64)).reshape(-1, 2)

        projected[frame] = [
            (candidate, (float(px), float(py)))
            for candidate, (px, py) in zip(found, pitch)
            if -margin_m <= px <= pitch_length + margin_m and -margin_m <= py <= pitch_width + margin_m
        ]

    # How many frames does each pitch cell contain a candidate?
    cell_frames = {}
    for frame, found in projected.items():
        for cell in {(int(px // cell_m), int(py // cell_m)) for _, (px, py) in found}:
            cell_frames[cell] = cell_frames.get(cell, 0) + 1

    static_cells = {cell for cell, count in cell_frames.items() if count >= static_frames}

    return {
        frame: [
            candidate for candidate, (px, py) in found
            if candidate[2] >= keep_confidence
            or (int(px // cell_m), int(py // cell_m)) not in static_cells
        ]
        for frame, found in projected.items()
    }


def drop_static_segments(points, homographies, min_travel_m=10.0):
    """Remove stretches that never go anywhere on the pitch.

    Pitch-side markings and the shoes of someone standing still give
    continuous, sometimes confident "ball" detections, but they stay within
    a few meters. The extent is measured between the 5th and 95th percentile
    of the positions so that one stray point does not save a static stretch.
    """

    by_segment = {}
    for point in points:
        if not point.interpolated and point.frame < len(homographies):
            by_segment.setdefault(point.segment, []).append(point)

    moving = set()

    for segment, members in by_segment.items():
        pitch = np.array([
            cv2.perspectiveTransform(
                np.array([[[p.x, p.y]]], dtype=np.float64),
                np.asarray(homographies[p.frame], dtype=np.float64),
            )[0, 0]
            for p in members
        ])

        low, high = np.percentile(pitch, [5, 95], axis=0)
        if float(np.linalg.norm(high - low)) >= min_travel_m:
            moving.add(segment)

    return [p for p in points if p.segment in moving]
