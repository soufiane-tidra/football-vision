import csv

from src.tracking.player import PlayerTrack


# Classes that are people. "person" is the generic COCO model; the others
# come from the football-specific detector.
PERSON_CLASSES = {"person", "player", "goalkeeper", "referee"}

# Roles that count as players in the analytics.
PLAYER_ROLES = {"person", "player", "goalkeeper"}


def load_player_tracks(csv_path, use_feet=True, roles=PLAYER_ROLES, max_frames=None):
    """Load player tracks from tracks.csv.

    A track's role is the majority class over all its detections, so a
    referee wrongly detected as "player" on a few frames stays a referee.
    Only tracks whose role is in `roles` are returned (None = everyone).

    use_feet=True stores the bottom-center of each box (the ground contact
    point), which is what the pitch homography needs. use_feet=False keeps
    the box center (old behaviour).
    """

    tracks = {}

    with open(csv_path, "r", newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["class_name"] not in PERSON_CLASSES:
                continue

            frame = int(row["frame"])

            if max_frames is not None and frame >= max_frames:
                continue

            track_id = int(row["track_id"])

            if track_id not in tracks:
                tracks[track_id] = PlayerTrack(track_id=track_id)

            x = float(row["center_x"])
            y = float(row["y2"]) if use_feet else float(row["center_y"])

            tracks[track_id].add_detection(
                frame=frame,
                x=x,
                y=y,
                confidence=float(row["confidence"]),
                class_name=row["class_name"],
                box=tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            )

    if roles is None:
        return tracks

    return {track_id: track for track_id, track in tracks.items() if track.role in roles}
