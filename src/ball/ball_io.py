"""Read ball.csv."""

import csv


def load_ball(path):
    """Return {frame: {"x", "y", "pitch_x", "pitch_y", "interpolated"}} (pitch values may be None)."""

    ball = {}

    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            ball[int(row["frame"])] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "pitch_x": float(row["pitch_x"]) if row["pitch_x"] else None,
                "pitch_y": float(row["pitch_y"]) if row["pitch_y"] else None,
                "interpolated": row["interpolated"] == "1",
            }

    return ball
