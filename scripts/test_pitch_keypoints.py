"""Show what the pitch keypoint model sees on one frame.

Draws the detected keypoints (green = confident, red = ignored) and the pitch
lines predicted from them, and prints the homography quality.

Run:
    python -m scripts.test_pitch_keypoints                  (frame 0)
    python -m scripts.test_pitch_keypoints --frame 600
    python -m scripts.test_pitch_keypoints --video other.mp4 --frame 100

Output:
    outputs/keypoints/frame_XXXXXX.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.pitch.drawing import draw_pitch_overlay
from src.pitch.keypoint_detector import PitchKeypointDetector
from src.pitch.keypoints import KEYPOINT_NAMES, homography_from_keypoints
from src.pitch.landmarks import PitchDimensions
from src.utils.config import load_config
from src.video.frames import read_frame


def main():
    config = load_config()
    settings = config.pitch_keypoints

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=config.video)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--weights", default=settings.weights)
    args = parser.parse_args()

    dims = PitchDimensions(length=config.pitch.length, width=config.pitch.width)
    detector = PitchKeypointDetector(args.weights, dims, imgsz=settings.imgsz)

    frame = read_frame(args.video, args.frame)
    xy, conf = detector.detect(frame)

    if xy is None:
        print("No pitch detected in this frame.")
        return

    mapper = homography_from_keypoints(
        xy, conf, detector.pitch_xy,
        min_confidence=settings.confidence,
        min_points=settings.min_points,
        max_error_m=settings.max_error_m,
    )

    output = draw_pitch_overlay(frame, mapper) if mapper is not None else frame.copy()

    confident = conf >= settings.confidence
    for i in range(len(xy)):
        if xy[i].sum() <= 0:
            continue
        color = (0, 255, 0) if confident[i] else (0, 0, 255)
        center = (int(xy[i][0]), int(xy[i][1]))
        cv2.circle(output, center, 7, color, -1)
        cv2.putText(output, f"{i + 1}", (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    print(f"Frame {args.frame}: {confident.sum()} confident keypoints (>= {settings.confidence})")
    for i in np.where(confident)[0]:
        print(f"  {i + 1:2d} {KEYPOINT_NAMES[i]:38s} conf {conf[i]:.2f}  at ({xy[i][0]:.0f}, {xy[i][1]:.0f})")

    if mapper is None:
        print("Homography: REJECTED (too few / collinear / inconsistent keypoints)")
    else:
        errors = mapper.reprojection_errors()[mapper.inlier_mask]
        print(f"Homography: OK | inliers {mapper.inlier_mask.sum()} | mean error {errors.mean():.2f} m")

    path = Path(config.paths.outputs) / "keypoints" / f"frame_{args.frame:06d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
