"""Collect ball candidates on every frame (low confidence, no tracker)."""

import cv2


def detect_ball_candidates(video_path, weights, imgsz=1280, confidence=0.05, max_frames=None):
    """Return {frame: [(x, y, confidence, box_size), ...]} with box centers in pixels.

    The confidence threshold is deliberately low: the ball is a few pixels
    wide and often scores poorly. False positives are removed afterwards by
    requiring a physically consistent trajectory (src.ball.trajectory).
    """

    # Imported here so the rest of the package works without torch.
    from ultralytics import YOLO

    model = YOLO(str(weights))
    ball_ids = [i for i, name in model.names.items() if name == "ball"]

    if not ball_ids:
        raise ValueError(f"{weights} has no 'ball' class (classes: {list(model.names.values())}).")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    candidates = {}
    frame_number = 0

    while max_frames is None or frame_number < max_frames:
        success, frame = video.read()
        if not success:
            break

        result = model(frame, imgsz=imgsz, conf=confidence, classes=ball_ids, verbose=False)[0]

        found = []
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            found.append((float((x1 + x2) / 2), float((y1 + y2) / 2), float(conf), float(max(x2 - x1, y2 - y1))))

        candidates[frame_number] = found
        frame_number += 1

    video.release()

    return candidates
