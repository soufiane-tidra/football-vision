"""Read players.csv (identified, stitched players in pitch coordinates)."""

import csv
from dataclasses import dataclass

import numpy as np

from src.tracking.player import PlayerTrack


@dataclass(frozen=True)
class PlayerInfo:
    player_id: int
    team: str          # "team_1", "team_2" or "" for officials
    role: str          # "player", "goalkeeper" or "referee"


def load_players(path):
    """Return (player_id -> PlayerTrack with pitch_positions, player_id -> PlayerInfo)."""

    tracks, info = {}, {}

    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            player_id = int(row["player_id"])

            if player_id not in tracks:
                tracks[player_id] = PlayerTrack(track_id=player_id)
                info[player_id] = PlayerInfo(player_id, row["team"], row["role"])

            box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            tracks[player_id].add_detection(
                int(row["frame"]), (box[0] + box[2]) / 2, box[3], 1.0, row["role"], box
            )

            if row["pitch_x"] and row["pitch_y"]:
                tracks[player_id].pitch_positions.append((float(row["pitch_x"]), float(row["pitch_y"])))
            else:
                tracks[player_id].pitch_positions.append((np.nan, np.nan))

    return tracks, info
