"""Validate the pitch calibration.

For every keyframe it prints:
    - reprojection error of each landmark (fit on all points)
    - leave-one-out error (fit without the point, then predict it)
      -> an honest estimate of real-world accuracy
and saves visual checks to outputs/calibration/:
    - keyframe_XXXXXX.jpg : predicted pitch lines drawn on the video frame
    - pitch_XXXXXX.jpg    : 2D pitch with true (green) vs mapped (red) landmarks

Run:
    python -m scripts.test_pitch_mapper
"""

from pathlib import Path

import cv2
import numpy as np

from src.pitch.calibration import Calibration
from src.pitch.drawing import PitchDiagram, draw_pitch_overlay
from src.pitch.homography import PitchMapper
from src.utils.config import load_config
from src.video.frames import read_frame


config = load_config()
CALIBRATION_PATH = config.paths.calibration
OUTPUT_DIR = Path(config.paths.outputs) / "calibration"


def leave_one_out_errors(points):
    errors = []

    for i in range(len(points)):
        train = points[:i] + points[i + 1:]

        if len(train) < 4:
            return None

        mapper = PitchMapper([p.image for p in train], [p.pitch for p in train])
        predicted = mapper.transform_points([points[i].image])[0]
        errors.append(float(np.linalg.norm(predicted - np.array(points[i].pitch))))

    return np.array(errors)


def main():
    calibration = Calibration.load(CALIBRATION_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Calibration: {CALIBRATION_PATH}")
    print(f"Pitch: {calibration.pitch.length} x {calibration.pitch.width} m")
    print(f"Keyframes: {[k.frame for k in calibration.keyframes]}")

    for keyframe in calibration.keyframes:
        mapper = keyframe.build_mapper()
        errors = mapper.reprojection_errors()
        loo = leave_one_out_errors(keyframe.points)
        mapped = mapper.transform_points([p.image for p in keyframe.points])

        print()
        print(f"=== Keyframe {keyframe.frame} ({len(keyframe.points)} points) ===")
        print(f"{'landmark':32s} {'image':>12s} {'true (m)':>14s} {'mapped (m)':>14s} {'err':>6s} {'LOO':>6s}")

        for i, p in enumerate(keyframe.points):
            loo_text = f"{loo[i]:6.2f}" if loo is not None else "   n/a"
            print(
                f"{p.landmark:32s} "
                f"({p.image[0]:4.0f},{p.image[1]:4.0f}) "
                f"({p.pitch[0]:5.1f},{p.pitch[1]:5.1f}) "
                f"({mapped[i][0]:5.1f},{mapped[i][1]:5.1f}) "
                f"{errors[i]:6.2f} {loo_text}"
            )

        print(f"Reprojection error : mean {errors.mean():.2f} m | max {errors.max():.2f} m")

        if loo is not None:
            print(f"Leave-one-out error: mean {loo.mean():.2f} m | max {loo.max():.2f} m")
            verdict = "GOOD" if loo.mean() < 0.5 else "OK" if loo.mean() < 1.0 else "POOR - re-click points"
            print(f"Verdict: {verdict}")
        else:
            print("Add a 5th point to get a leave-one-out accuracy estimate.")

        outliers = [p.landmark for p, ok in zip(keyframe.points, mapper.inlier_mask) if not ok]
        if outliers:
            print(f"RANSAC rejected: {outliers}")

        # Visual checks
        frame = read_frame(calibration.video_path, keyframe.frame)
        overlay = draw_pitch_overlay(frame, mapper, calibration.pitch)

        for p in keyframe.points:
            cv2.circle(overlay, (int(p.image[0]), int(p.image[1])), 6, (0, 255, 0), -1)

        overlay_path = OUTPUT_DIR / f"keyframe_{keyframe.frame:06d}.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        diagram = PitchDiagram(calibration.pitch)
        image = diagram.draw_points([p.pitch for p in keyframe.points], (0, 255, 0), radius=7)
        image = diagram.draw_points(mapped, (0, 0, 255), radius=3, image=image)

        pitch_path = OUTPUT_DIR / f"pitch_{keyframe.frame:06d}.jpg"
        cv2.imwrite(str(pitch_path), image)

        print(f"Saved: {overlay_path}")
        print(f"Saved: {pitch_path}")


if __name__ == "__main__":
    main()
