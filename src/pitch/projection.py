"""Convert tracked image positions to pitch coordinates (meters)."""

import numpy as np

from src.pitch.homography import PitchMapper
from src.pitch.landmarks import PitchDimensions


def project_tracks_to_pitch(players, homographies, dims=PitchDimensions(), margin_m=3.0):
    """Fill player.pitch_positions using the per-frame homographies.

    Positions further than margin_m outside the pitch (staff, cameramen,
    crowd) are set to NaN.
    """

    mappers = {}

    for player in players.values():
        pitch_positions = []

        for frame, (x, y) in zip(player.frames, player.positions):

            if not 0 <= frame < len(homographies):
                pitch_positions.append((np.nan, np.nan))
                continue

            if frame not in mappers:
                mappers[frame] = PitchMapper.from_matrix(homographies[frame])

            px, py = mappers[frame].transform_point(x, y)

            on_pitch = (
                -margin_m <= px <= dims.length + margin_m
                and -margin_m <= py <= dims.width + margin_m
            )

            pitch_positions.append((float(px), float(py)) if on_pitch else (np.nan, np.nan))

        player.pitch_positions = pitch_positions

    return players


def on_pitch_ratio(player):
    if not player.pitch_positions:
        return 0.0

    points = np.array(player.pitch_positions)
    return float((~np.isnan(points).any(axis=1)).mean())
