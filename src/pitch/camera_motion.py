"""Per-frame pitch homographies for a panning / zooming camera.

Only a few keyframes are calibrated by hand. For every other frame:

1. Predict: estimate camera motion M (previous frame -> current frame) with
   sparse optical flow on the pitch area only, then H_pred = H_prev . M^-1
2. Refine: render the pitch lines predicted by H_pred and align them to the
   white lines detected in the frame (ECC, homography model). This corrects
   the small errors that would otherwise accumulate (drift).

Everything runs at a reduced resolution for speed; homographies are returned
for full-resolution pixels (image -> pitch meters).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.pitch.calibration import Calibration
from src.pitch.drawing import (
    densify,
    draw_pitch_lines,
    get_dense_pitch_lines,
    project_pitch_to_image,
)
from src.pitch.landmarks import PitchDimensions


logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    scale: float = 0.5               # processing resolution
    ignore_top: float = 0.06         # static overlays (title bar), fraction of height
    ignore_bottom: float = 0.12      # static overlays (player controls)
    pitch_margin_m: float = 2.0      # area around the pitch used for features/lines
    max_corners: int = 600
    refine: bool = True
    ecc_scale: float = 0.5           # ECC runs at this fraction of the processing resolution (speed)
    ecc_iterations: int = 30
    ecc_min_correlation: float = 0.35
    ecc_max_shift_px: float = 20.0   # reject refinements that move the image more than this


@dataclass
class TrackingStats:
    frames: int = 0
    motion_failures: int = 0
    refined: int = 0
    refine_rejected: int = 0


class PitchHomographyTracker:

    def __init__(self, dims: PitchDimensions = PitchDimensions(), config: TrackerConfig = TrackerConfig()):
        self.dims = dims
        self.config = config
        self.dense_lines = get_dense_pitch_lines(dims, step=1.0)

        s = config.scale
        self.S = np.diag([s, s, 1.0])          # full -> small pixels
        self.S_inv = np.diag([1 / s, 1 / s, 1.0])

        m = config.pitch_margin_m
        outline = np.array([
            [-m, -m], [dims.length + m, -m],
            [dims.length + m, dims.width + m], [-m, dims.width + m], [-m, -m],
        ])
        self.pitch_outline = densify(outline, step=1.0)

        self.overlay_mask = None
        self.stats = TrackingStats()

    # ---------- public ----------

    def process(self, video_path, calibration: Calibration, max_frames=None):
        """Return an (N, 3, 3) array of full-resolution image -> pitch homographies.

        Frames between two keyframes are tracked forward from the left one and
        backward from the right one, then blended by distance, so the error
        stays small everywhere and there is no jump at keyframes. Frames after
        the last keyframe are tracked forward only (streamed, no buffering).
        """

        if not calibration.keyframes:
            raise ValueError("Calibration has no keyframes.")

        keyframes = {
            k.frame: k.build_mapper().matrix @ self.S_inv
            for k in calibration.keyframes
            if max_frames is None or k.frame < max_frames
        }

        if not keyframes:
            raise ValueError(f"No keyframe before frame {max_frames}.")

        last_keyframe = max(keyframes)

        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        homographies = {}      # frame -> small-image -> pitch
        buffer = []            # (frame, prepared) since the previous keyframe
        left = None            # (frame, prepared) of the previous keyframe
        previous = None
        H = None
        t = 0

        while max_frames is None or t < max_frames:
            success, frame = video.read()
            if not success:
                break

            current = self._prepare(frame)

            if t in keyframes:
                self._solve_interval(left, buffer, (t, current), keyframes, homographies)
                homographies[t] = keyframes[t]
                left, buffer = (t, current), []
                H = keyframes[t]
            elif t > last_keyframe:
                H = self._step(previous, current, H)
                homographies[t] = H
            else:
                buffer.append((t, current))

            previous = current
            t += 1

        video.release()

        if last_keyframe >= t:
            raise ValueError(f"Keyframe {last_keyframe} is outside the video (0-{t - 1}).")

        self.stats.frames = t

        return np.array([self._normalize(homographies[i] @ self.S) for i in range(t)])

    def _solve_interval(self, left, buffer, right, keyframes, homographies):
        """Fill the frames in `buffer`, lying between keyframes `left` and `right`."""

        if not buffer:
            return

        forward = {}
        if left is not None:
            H, previous = keyframes[left[0]], left[1]
            for t, prepared in buffer:
                H = self._step(previous, prepared, H)
                forward[t] = H
                previous = prepared

        backward = {}
        H, following = keyframes[right[0]], right[1]
        for t, prepared in reversed(buffer):
            H = self._step(following, prepared, H)
            backward[t] = H
            following = prepared

        for t, _ in buffer:
            if left is None:
                homographies[t] = backward[t]
            else:
                weight = (t - left[0]) / (right[0] - left[0])
                homographies[t] = _blend(forward[t], backward[t], weight)

    # ---------- per-frame ----------

    def _prepare(self, frame):
        small = cv2.resize(
            frame, None, fx=self.config.scale, fy=self.config.scale, interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.overlay_mask is None:
            h = gray.shape[0]
            self.overlay_mask = np.full(gray.shape, 255, dtype=np.uint8)
            self.overlay_mask[: int(h * self.config.ignore_top)] = 0
            self.overlay_mask[int(h * (1 - self.config.ignore_bottom)):] = 0

        return {"gray": gray, "lines": self._detect_lines(small, gray)}

    def _step(self, previous, current, H_previous):
        motion = self._estimate_motion(previous["gray"], current["gray"], H_previous)

        if motion is None:
            self.stats.motion_failures += 1
            motion = np.eye(3)

        H = self._normalize(H_previous @ np.linalg.inv(motion))

        if self.config.refine:
            H = self._refine(current["lines"], H)

        return H

    def _pitch_mask(self, H):
        """Binary mask of the (slightly enlarged) pitch area in the image."""

        polygon = project_pitch_to_image(np.linalg.inv(H), self.pitch_outline)
        polygon = polygon[~np.isnan(polygon).any(axis=1)]

        mask = np.zeros_like(self.overlay_mask)

        if len(polygon) >= 3:
            polygon = np.clip(polygon, -1e4, 1e4).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)

        return cv2.bitwise_and(mask, self.overlay_mask)

    def _estimate_motion(self, previous_gray, gray, H_previous):
        mask = self._pitch_mask(H_previous)

        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=self.config.max_corners,
            qualityLevel=0.005, minDistance=7, mask=mask,
        )

        if points is None or len(points) < 20:
            # Not enough texture on the pitch: fall back to the whole frame.
            points = cv2.goodFeaturesToTrack(
                previous_gray, maxCorners=self.config.max_corners,
                qualityLevel=0.01, minDistance=7, mask=self.overlay_mask,
            )

        if points is None or len(points) < 20:
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3
        )

        good = status.ravel() == 1
        if good.sum() < 20:
            return None

        # RANSAC rejects moving players.
        motion, _ = cv2.findHomography(points[good], next_points[good], cv2.RANSAC, 1.0)

        return motion

    def _detect_lines(self, small_bgr, gray):
        """White, thin, unsaturated structures = pitch markings."""

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        saturation = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)[..., 1]

        lines = ((tophat > 18) & (saturation < 90)).astype(np.uint8) * 255

        return cv2.bitwise_and(lines, self.overlay_mask)

    def _refine(self, observed_lines, H_pred):
        """Align predicted pitch lines with observed lines (ECC homography)."""

        cfg = self.config

        observed = cv2.bitwise_and(observed_lines, self._pitch_mask(H_pred))

        rendered = np.zeros_like(observed)
        draw_pitch_lines(rendered, np.linalg.inv(H_pred), self.dense_lines, 255, 2, cv2.LINE_8)

        k = cfg.ecc_scale
        template = cv2.resize(
            cv2.GaussianBlur(observed, (0, 0), 3), None, fx=k, fy=k, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        moving = cv2.resize(
            cv2.GaussianBlur(rendered, (0, 0), 3), None, fx=k, fy=k, interpolation=cv2.INTER_AREA
        ).astype(np.float32)

        warp = np.eye(3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_iterations, 1e-5)

        try:
            correlation, warp = cv2.findTransformECC(
                template, moving, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
            )
        except cv2.error:
            self.stats.refine_rejected += 1
            return H_pred

        # warp maps observed pixels -> predicted-render pixels; bring it back
        # from the ECC resolution to the processing resolution.
        K = np.diag([k, k, 1.0])
        warp = np.linalg.inv(K) @ warp.astype(np.float64) @ K

        h, w = observed.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), warp).reshape(-1, 2)
        shift = np.linalg.norm(moved - corners, axis=1).max()

        if correlation < cfg.ecc_min_correlation or shift > cfg.ecc_max_shift_px:
            self.stats.refine_rejected += 1
            return H_pred

        self.stats.refined += 1
        return self._normalize(H_pred @ warp)

    @staticmethod
    def _normalize(H):
        # Not H / H[2, 2]: H[2, 2] passes through zero when the image corner
        # crosses the horizon, which makes the matrix explode.
        return H / np.linalg.norm(H)


def _blend(A, B, weight):
    """Interpolate two nearby homographies (weight 0 -> A, 1 -> B)."""

    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    # Homographies are defined up to sign: align them before averaging.
    if np.sum(A * B) < 0:
        B = -B

    return (1 - weight) * A + weight * B


def save_homographies(path, homographies):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, homographies)


def load_homographies(path):
    return np.load(path)
