"""Compute one image -> pitch homography per video frame.

The camera pans and zooms, so the keyframe calibration is propagated to every
frame using camera-motion estimation + pitch-line refinement.

Run:
    python -m scripts.compute_homographies
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

from src.pitch.calibration import Calibration
from src.pitch.camera_motion import PitchHomographyTracker, TrackerConfig, save_homographies
from src.pitch.drawing import draw_pitch_overlay
from src.pitch.homography import PitchMapper
from src.utils.config import load_config


def parse_args():
    config = load_config()

    parser = argparse.ArgumentParser()
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    calibration = Calibration.load(args.calibration)
    print(f"Keyframes: {[k.frame for k in calibration.keyframes]}")

    tracker = PitchHomographyTracker(
        calibration.pitch, TrackerConfig(refine=not args.no_refine)
    )

    print("Tracking the pitch through the video (about 3 minutes)...")
    start = time.time()
    homographies = tracker.process(calibration.video_path, calibration)

    stats = tracker.stats
    print(f"Frames: {stats.frames} in {time.time() - start:.0f}s")
    print(f"Line refinement: {stats.refined} accepted | {stats.refine_rejected} rejected")
    if stats.motion_failures:
        print(f"WARNING: camera motion failed on {stats.motion_failures} frames")

    save_homographies(args.output, homographies)
    print(f"Saved: {args.output}")

    if not args.no_video:
        print("Writing overlay check video...")
        write_overlay_video(calibration.video_path, homographies, calibration.pitch, args.overlay)
        print(f"Saved: {args.overlay}")
        print("Open it: the red lines should stay on the real pitch lines all the way through.")


if __name__ == "__main__":
    main()
