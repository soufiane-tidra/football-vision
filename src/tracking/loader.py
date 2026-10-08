import csv

from src.tracking.player import PlayerTrack


def load_player_tracks(csv_path, use_feet=True):
    """Load 'person' tracks from tracks.csv.

    use_feet=True stores the bottom-center of each box (the ground contact
    point), which is what the pitch homography needs. use_feet=False keeps
    the box center (old behaviour).
    """

    players = {}

    with open(csv_path, "r", newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["class_name"] != "person":
                continue

            track_id = int(row["track_id"])

            if track_id not in players:
                players[track_id] = PlayerTrack(track_id=track_id)

            x = float(row["center_x"])
            y = float(row["y2"]) if use_feet else float(row["center_y"])

            players[track_id].add_detection(
                frame=int(row["frame"]),
                x=x,
                y=y,
                confidence=float(row["confidence"])
            )

    return players
