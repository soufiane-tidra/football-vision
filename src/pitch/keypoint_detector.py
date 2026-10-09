"""Per-frame pitch calibration with a YOLO-pose pitch keypoint model."""

import numpy as np

from src.pitch.keypoints import NUM_KEYPOINTS, homography_from_keypoints, pitch_keypoints
from src.pitch.landmarks import PitchDimensions


class PitchKeypointDetector:

    def __init__(
        self,
        weights,
        dims: PitchDimensions = PitchDimensions(),
        confidence=0.5,
        min_points=4,
        max_error_m=1.0,
        imgsz=640,
    ):
        # Imported here so the rest of the package works without torch.
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.pitch_xy = pitch_keypoints(dims)
        self.confidence = confidence
        self.min_points = min_points
        self.max_error_m = max_error_m
        self.imgsz = imgsz

    def detect(self, frame):
        """Return (image_xy (32, 2), confidence (32,)) for the best pitch detection, or (None, None)."""

        result = self.model(frame, imgsz=self.imgsz, verbose=False)[0]
        keypoints = result.keypoints

        if keypoints is None or len(keypoints) == 0:
            return None, None

        best = 0
        if result.boxes is not None and len(result.boxes):
            best = int(result.boxes.conf.argmax())

        xy = keypoints.xy[best].cpu().numpy()

        if len(xy) != NUM_KEYPOINTS:
            raise ValueError(
                f"Model predicts {len(xy)} keypoints, expected {NUM_KEYPOINTS}. Wrong weights?"
            )

        if keypoints.conf is None:
            conf = np.ones(len(xy))
        else:
            conf = keypoints.conf[best].cpu().numpy().copy()

        # Off-screen keypoints come back clamped to the image border, often with
        # high confidence: they are not real detections.
        h, w = frame.shape[:2]
        margin = 2
        on_border = (
            (xy[:, 0] <= margin) | (xy[:, 1] <= margin)
            | (xy[:, 0] >= w - 1 - margin) | (xy[:, 1] >= h - 1 - margin)
        )
        conf[on_border] = 0.0

        return xy, conf

    def mapper(self, frame):
        """PitchMapper for this frame, or None if the detection is not reliable."""

        xy, conf = self.detect(frame)

        if xy is None:
            return None

        return homography_from_keypoints(
            xy, conf, self.pitch_xy,
            min_confidence=self.confidence,
            min_points=self.min_points,
            max_error_m=self.max_error_m,
        )
