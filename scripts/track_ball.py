"""Track the ball through the video.

    low-confidence ball candidates on every frame
        -> best physically consistent trajectory (dynamic programming)
        -> short gaps interpolated
        -> ball.csv with pixel and pitch coordinates

Requires:
    models/football_detector.pt  (python -m scripts.train_detector)
    python -m scripts.compute_homographies --method keypoints

Run:
    python -m scripts.track_ball

Note: pitch coordinates assume the ball is on the ground. While it is in the
air the projected position is shifted away from the camera.
"""

import csv
from pathlib import Path

from src.ball.detector import detect_ball_candidates
from src.ball.trajectory import drop_static_segments, filter_candidates, interpolate_gaps, select_trajectory
from src.pitch.camera_motion import load_homographies
from src.pitch.homography import PitchMapper
from src.utils.config import load_config
from src.video.video import get_video_info


def main():
    config = load_config()
    settings = config.ball
    fps = get_video_info(config.video)["fps"]

    print("1/3 Detecting ball candidates...")
    candidates = detect_ball_candidates(
        config.video, config.player_detection.weights,
        imgsz=config.detection.imgsz, confidence=settings.confidence,
    )
    n_frames = len(candidates)
    total = sum(len(found) for found in candidates.values())

    homographies = load_homographies(config.paths.homographies)

    print("2/3 Removing markings / off-pitch candidates, selecting the trajectory...")
    candidates = filter_candidates(
        candidates, homographies, config.pitch.length, config.pitch.width,
        static_frames=int(settings.static_s * fps),
    )
    kept = sum(len(found) for found in candidates.values())
    with_candidates = sum(1 for found in candidates.values() if found)
    detected = select_trajectory(
        candidates, max_speed_px=settings.max_speed_px, max_link_gap=settings.max_gap_frames,
        min_segment_confidence=settings.min_segment_confidence,
    )
    detected = drop_static_segments(detected, homographies, settings.min_travel_m)
    points = interpolate_gaps(detected, max_gap=settings.max_gap_frames)

    print("3/3 Projecting to the pitch...")

    output = Path(config.paths.ball)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x", "y", "confidence", "interpolated", "segment", "pitch_x", "pitch_y"])

        for p in points:
            pitch_x = pitch_y = ""
            if p.frame < len(homographies):
                px, py = PitchMapper.from_matrix(homographies[p.frame]).transform_point(p.x, p.y)
                pitch_x, pitch_y = f"{px:.2f}", f"{py:.2f}"

            writer.writerow([
                p.frame, f"{p.x:.1f}", f"{p.y:.1f}", f"{p.confidence:.3f}",
                int(p.interpolated), p.segment, pitch_x, pitch_y,
            ])

    interpolated = sum(p.interpolated for p in points)

    print()
    print(f"Candidates: {total} -> {kept} after filtering | frames with candidates: "
          f"{with_candidates}/{n_frames} ({with_candidates / n_frames:.0%})")
    print(f"Ball located in {len(points)} frames ({len(points) / n_frames:.0%}): "
          f"{len(detected)} detected + {interpolated} interpolated")
    print(f"Continuous segments: {len({p.segment for p in points})}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
