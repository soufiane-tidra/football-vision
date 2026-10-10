"""Turn raw tracker fragments into identified players.

    tracks.csv (fragments)  +  homographies  +  video (jersey colors)
        -> team and role per track (color clusters + detector vote + position)
        -> fragments of the same person stitched together
        -> players.csv : one row per player per frame, in pitch meters

Requires:
    python -m scripts.track
    python -m scripts.compute_homographies --method keypoints

Run:
    python -m scripts.build_players
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.classification.identity import GOALKEEPER, PLAYER, REFEREE, assign_identities, is_pitch_side_staff
from src.classification.team_classifier import collect_track_colors
from src.pitch.camera_motion import load_homographies
from src.pitch.landmarks import PitchDimensions
from src.pitch.projection import on_pitch_ratio, project_tracks_to_pitch
from src.tracking.loader import load_player_tracks
from src.tracking.stitching import stitch_tracks
from src.utils.config import load_config
from src.video.video import get_video_info


def team_label(identity):
    return "" if identity.team is None else f"team_{identity.team + 1}"


def main():
    config = load_config()
    settings = config.stitching
    dims = PitchDimensions(length=config.pitch.length, width=config.pitch.width)
    fps = get_video_info(config.video)["fps"]

    print("1/4 Loading tracks and projecting to the pitch...")
    tracks = load_player_tracks(config.paths.tracks, roles=None)
    project_tracks_to_pitch(tracks, load_homographies(config.paths.homographies), dims)

    on_pitch = {t: track for t, track in tracks.items() if on_pitch_ratio(track) >= 0.5}
    print(f"    fragments: {len(tracks)} | on the pitch: {len(on_pitch)}")

    print("2/4 Sampling jersey colors...")
    colors = collect_track_colors(config.video, on_pitch)

    print("3/4 Assigning teams and roles...")
    identities, _ = assign_identities(on_pitch, colors, dims)

    # Coaches / cameramen at the touchline are not players; officials may stand there.
    staff = {
        t for t, identity in identities.items()
        if identity.role != REFEREE
        and is_pitch_side_staff(on_pitch[t], dims, band_m=settings.touchline_band_m)
    }
    identified = {t: on_pitch[t] for t in identities if t not in staff}
    print(f"    ignored as pitch-side staff: {len(staff)} fragments")

    print("4/4 Stitching fragments...")
    merged, mapping = stitch_tracks(
        identified, identities, fps,
        max_gap_s=settings.max_gap_s,
        max_speed_mps=settings.max_speed_mps,
        slack_m=settings.slack_m,
    )

    min_detections = settings.min_player_s * fps
    kept = {t: track for t, track in merged.items() if track.detection_count >= min_detections}

    # Stable, readable ids: team 1, team 2, then officials; by first appearance.
    def order(track_id):
        identity = identities[track_id]
        return (identity.team if identity.team is not None else 9, kept[track_id].first_frame)

    player_ids = {old: new for new, old in enumerate(sorted(kept, key=order), start=1)}

    output = Path(config.paths.players)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "player_id", "team", "role", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y"])

        rows = []
        for old_id, track in kept.items():
            identity = identities[old_id]
            for frame, box, (px, py) in zip(track.frames, track.boxes, track.pitch_positions):
                rows.append([
                    frame, player_ids[old_id], team_label(identity), identity.role,
                    *(f"{v:.1f}" for v in box),
                    "" if np.isnan(px) else f"{px:.2f}",
                    "" if np.isnan(py) else f"{py:.2f}",
                ])

        writer.writerows(sorted(rows, key=lambda r: (r[0], r[1])))

    # ---------- summary ----------
    counts = Counter((team_label(identities[t]) or "officials", identities[t].role) for t in kept)
    per_frame = Counter(row[0] for row in rows if row[3] != REFEREE)
    people = np.array(list(per_frame.values()))
    coverage = [track.detection_count / (track.last_frame - track.first_frame + 1) for track in kept.values()]

    print()
    print(f"Fragments identified: {len(identified)} -> people after stitching: {len(merged)} -> kept: {len(kept)}")
    for (team, role), n in sorted(counts.items()):
        print(f"  {team:10s} {role:11s} {n}")
    print(f"Players + goalkeepers per frame: median {np.median(people):.0f} (min {people.min()}, max {people.max()})")
    print(f"Longest track: {max(t.detection_count for t in kept.values()) / fps:.1f} s | "
          f"mean detection coverage inside tracks: {np.mean(coverage):.0%}")
    print(f"Roles: {PLAYER}, {GOALKEEPER}, {REFEREE}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
