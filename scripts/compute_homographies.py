"""Compute one image -> pitch homography per video frame.

Two methods:
    manual     (default) hand-clicked keyframes (pitch_calibration.json),
               propagated with camera motion + pitch-line refinement
    keypoints  trained pitch keypoint model on every frame; frames without a
               reliable detection are filled by camera tracking, then the
               homographies are smoothed over time

Run:
    python -m scripts.compute_homographies
    python -m scripts.compute_homographies --method keypoints
    python -m scripts.compute_homographies --no-video     (skip the check video)
    python -m scripts.compute_homographies --no-refine    (motion only, for comparison)

Output:
    data/processed/homographies.npy           (N, 3, 3) image pixels -> pitch meters
    outputs/calibration/pitch_overlay.mp4     predicted pitch lines on every frame
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.pitch.calibration import Calibration, CalibrationPoint, Keyframe
from src.pitch.camera_motion import PitchHomographyTracker, TrackerConfig, save_homographies
from src.pitch.drawing import draw_pitch_overlay
from src.pitch.homography import PitchMapper
from src.pitch.keypoints import KEYPOINT_NAMES, pitch_keypoints, smooth_homographies
from src.pitch.landmarks import PitchDimensions
from src.utils.config import load_config


def parse_args():
    config = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["manual", "keypoints"], default="manual")
    parser.add_argument("--calibration", default=config.paths.calibration)
    parser.add_argument("--output", default=config.paths.homographies)
    parser.add_argument("--overlay", default=str(Path(config.paths.outputs) / "calibration" / "pitch_overlay.mp4"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-refine", action="store_true")
    return parser.parse_args()


def write_overlay_video(video_path, homographies, dims, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for index, H in enumerate(homographies):
        success, frame = video.read()
        if not success:
            break

        overlay = draw_pitch_overlay(frame, PitchMapper.from_matrix(H), dims)
        cv2.putText(overlay, f"frame {index}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        writer.write(overlay)

    video.release()
    writer.release()


def detect_keypoint_calibration(config):
    """Run the pitch keypoint model on every frame; reliable frames become keyframes."""

    from src.pitch.keypoint_detector import PitchKeypointDetector

    settings = config.pitch_keypoints
    dims = PitchDimensions(length=config.pitch.length, width=config.pitch.width)

    if not Path(settings.weights).exists():
        raise SystemExit(f"{settings.weights} not found. Run: python -m scripts.train_pitch_keypoints")

    detector = PitchKeypointDetector(
        settings.weights, dims,
        confidence=settings.confidence,
        min_points=settings.min_points,
        max_error_m=settings.max_error_m,
        imgsz=settings.imgsz,
    )
    keypoints_xy = pitch_keypoints(dims)

    def name_of(pitch_point):
        # Mapper points are float32: match by nearest keypoint, not exact equality.
        return KEYPOINT_NAMES[int(np.argmin(np.linalg.norm(keypoints_xy - pitch_point, axis=1)))]

    calibration = Calibration(video_path=config.video, pitch=dims)
    video = cv2.VideoCapture(config.video)
    frame_number = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        mapper = detector.mapper(frame)

        if mapper is not None:
            inliers = mapper.inlier_mask
            points = [
                CalibrationPoint(name_of(p), tuple(i), tuple(p))
                for i, p in zip(mapper.image_points[inliers].tolist(), mapper.pitch_points[inliers].tolist())
            ]
            calibration.set_keyframe(Keyframe(frame_number, points))

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"  {frame_number} frames | reliable: {len(calibration.keyframes)}")

    video.release()

    frames = [k.frame for k in calibration.keyframes]
    gaps = np.diff([-1] + frames + [frame_number]) - 1
    print(
        f"Reliable keypoint frames: {len(frames)}/{frame_number} ({len(frames) / max(frame_number, 1):.0%}) | "
        f"longest gap: {gaps.max()} frames"
    )

    if not frames:
        raise SystemExit("No reliable pitch detection in the video.")

    return calibration


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = load_config()

    if args.method == "keypoints":
        print("Detecting pitch keypoints on every frame...")
        calibration = detect_keypoint_calibration(config)
        auto_path = Path(config.paths.calibration).with_name("pitch_calibration_auto.json")
        calibration.save(auto_path)
        print(f"Saved: {auto_path}")
    else:
        calibration = Calibration.load(args.calibration)
        print(f"Keyframes: {[k.frame for k in calibration.keyframes]}")

    tracker = PitchHomographyTracker(
        calibration.pitch,
        TrackerConfig(
            refine=not args.no_refine,
            # Detected keypoints are ~2 m accurate: snap them onto the visible pitch lines.
            refine_keyframes=args.method == "keypoints" and not args.no_refine,
        ),
    )

    print("Tracking the pitch through the video (fills frames between keyframes)...")
    start = time.time()
    homographies = tracker.process(calibration.video_path, calibration)

    stats = tracker.stats
    print(f"Frames: {stats.frames} in {time.time() - start:.0f}s")
    print(f"Line refinement: {stats.refined} accepted | {stats.refine_rejected} rejected")
    if stats.motion_failures:
        print(f"WARNING: camera motion failed on {stats.motion_failures} frames")

    if args.method == "keypoints":
        window = config.pitch_keypoints.smoothing_frames
        homographies = smooth_homographies(homographies, np.ones(len(homographies), dtype=bool), window)

    save_homographies(args.output, homographies)
    print(f"Saved: {args.output}")

    if not args.no_video:
        print("Writing overlay check video...")
        write_overlay_video(calibration.video_path, homographies, calibration.pitch, args.overlay)
        print(f"Saved: {args.overlay}")
        print("Open it: the red lines should stay on the real pitch lines all the way through.")


if __name__ == "__main__":
    main()
