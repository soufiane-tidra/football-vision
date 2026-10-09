"""Inspect the pitch keypoint dataset before training.

- draws the labelled keypoints (index + name) on a few training images
- checks that the dataset orientation matches our pitch coordinate system:
    keypoint "corner_top_left" (y = 0) must be the FAR touchline (top of image)
    keypoints of the left half must be on the LEFT of the halfway line

Run:
    python -m scripts.check_pitch_dataset
Output:
    outputs/pitch_dataset_check/*.jpg
"""

from pathlib import Path

import cv2
import numpy as np

from src.pitch.keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS
from src.utils.config import load_config


SAMPLES = 8

LEFT_HALF = [i for i, n in enumerate(KEYPOINT_NAMES) if n.startswith("left_") or n.endswith("_left")]
HALFWAY = [
    KEYPOINT_NAMES.index(n)
    for n in ("halfway_top", "center_circle_top", "center_circle_bottom", "halfway_bottom")
]
TOP_BOTTOM_PAIRS = [
    (KEYPOINT_NAMES.index(f"{p}_top"), KEYPOINT_NAMES.index(f"{p}_bottom"))
    for p in ("halfway", "center_circle", "left_penalty_corner", "right_penalty_corner",
              "left_goal_area_corner", "right_goal_area_corner")
]


def read_label(path, width, height):
    """Return (32, 3) array of x_px, y_px, visible for the first object in a YOLO-pose label file."""

    values = path.read_text().split()
    if len(values) < 5 + NUM_KEYPOINTS * 2:
        return None

    raw = np.array(values[5:], dtype=float)
    dims = 3 if len(raw) == NUM_KEYPOINTS * 3 else 2
    raw = raw[: NUM_KEYPOINTS * dims].reshape(NUM_KEYPOINTS, dims)

    visible = raw[:, 2] > 0 if dims == 3 else (raw[:, 0] > 0) | (raw[:, 1] > 0)
    return np.column_stack([raw[:, 0] * width, raw[:, 1] * height, visible])


def main():
    config = load_config()
    dataset = Path(config.pitch_keypoints.dataset_dir)
    output = Path(config.paths.outputs) / "pitch_dataset_check"
    output.mkdir(parents=True, exist_ok=True)

    images = sorted((dataset / "train" / "images").glob("*"))
    if not images:
        raise SystemExit(f"No images in {dataset / 'train' / 'images'}. Run: python -m scripts.download_pitch_dataset")

    top_ok = top_total = side_ok = side_total = 0
    visible_counts = []

    for i, image_path in enumerate(images):
        label_path = dataset / "train" / "labels" / (image_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        kps = read_label(label_path, w, h)
        if kps is None:
            continue

        vis = kps[:, 2] > 0
        visible_counts.append(int(vis.sum()))

        # Orientation: "_top" points must be higher in the image than "_bottom" points.
        for top, bottom in TOP_BOTTOM_PAIRS:
            if vis[top] and vis[bottom]:
                top_total += 1
                top_ok += kps[top, 1] < kps[bottom, 1]

        # Left-half points must be left of the halfway line.
        halfway_x = [kps[j, 0] for j in HALFWAY if vis[j]]
        if halfway_x:
            for j in LEFT_HALF:
                if vis[j]:
                    side_total += 1
                    side_ok += kps[j, 0] < np.mean(halfway_x)

        if i < SAMPLES:
            for j in np.where(vis)[0]:
                x, y = int(kps[j, 0]), int(kps[j, 1])
                cv2.circle(image, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(image, f"{j + 1} {KEYPOINT_NAMES[j]}", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(output / f"{image_path.stem[:40]}.jpg"), image)

    print(f"Train images with labels: {len(visible_counts)}")
    if visible_counts:
        print(f"Visible keypoints per image: mean {np.mean(visible_counts):.1f} | min {min(visible_counts)}")

    def report(name, ok, total):
        ratio = ok / total if total else float("nan")
        status = "OK" if total and ratio > 0.9 else "CHECK"
        print(f"{name}: {ok}/{total} consistent ({ratio:.0%}) -> {status}")
        return status == "OK"

    vertical = report("Top/bottom orientation (y = 0 is the far touchline)", top_ok, top_total)
    horizontal = report("Left/right orientation (x = 0 is the left goal)", side_ok, side_total)

    print()
    print(f"Saved sample images to: {output}")
    if vertical and horizontal:
        print("Orientation matches our pitch coordinates. Next: python -m scripts.train_pitch_keypoints")
    else:
        print("Orientation does NOT match - share this output before training.")


if __name__ == "__main__":
    main()
