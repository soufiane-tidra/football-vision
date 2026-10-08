"""Interactive pitch calibration.

Usage:
    python -m scripts.calibrate_pitch                 (frame 0)
    python -m scripts.calibrate_pitch --frame 600     (add another keyframe)

Workflow:
    1. Pick the landmark to place with N / P (it is highlighted on the "Pitch Map" window).
    2. Click exactly where that landmark is in the video frame (use the magnifier).
    3. Repeat for every landmark you can see clearly (6+ recommended, spread out).
    4. Red lines = pitch lines predicted by your calibration. They must sit on the real lines.
    5. Press S to save.

Keys:
    N / P        next / previous landmark
    Left click   place current landmark
    Right click  delete nearest point
    C            custom point (type X Y meters in the terminal, then click)
    U            undo last point
    O            toggle predicted pitch-line overlay
    S            save keyframe and exit
    Q / ESC      quit without saving
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.pitch.calibration import Calibration, CalibrationPoint, Keyframe
from src.pitch.drawing import PitchDiagram, draw_pitch_overlay
from src.pitch.homography import PitchMapper
from src.pitch.landmarks import PitchDimensions, get_landmarks
from src.utils.config import load_config
from src.video.frames import read_frame


MAX_DISPLAY_WIDTH = 1280
MAGNIFIER_RADIUS = 40
MAGNIFIER_ZOOM = 4

KEY_LEFT = (2424832, 65361)
KEY_RIGHT = (2555904, 65363)

WINDOW = "Pitch Calibration"
MAP_WINDOW = "Pitch Map"


def parse_args():
    config = load_config()

    parser = argparse.ArgumentParser(description="Manual multi-landmark pitch calibration.")
    parser.add_argument("--video", default=config.video)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", default=config.paths.calibration)
    parser.add_argument("--length", type=float, default=config.pitch.length)
    parser.add_argument("--width", type=float, default=config.pitch.width)
    return parser.parse_args()


class CalibrationTool:

    def __init__(self, frame, frame_number, dims, existing_points):
        self.frame = frame
        self.frame_number = frame_number
        self.dims = dims

        self.landmarks = get_landmarks(dims)
        self.landmark_names = list(self.landmarks)
        self.current = 0

        self.points: list[CalibrationPoint] = list(existing_points)
        self.custom_pending: tuple[float, float] | None = None

        self.scale = min(1.0, MAX_DISPLAY_WIDTH / frame.shape[1])
        self.mouse = (0, 0)
        self.show_overlay = True

        self.mapper = None
        self.overlay_cache = None
        self.diagram = PitchDiagram(dims, scale=7)

        self.update_mapper()

    # ---------- state ----------

    @property
    def current_name(self):
        return self.landmark_names[self.current]

    def update_mapper(self):
        self.mapper = None
        self.overlay_cache = None

        if len(self.points) < 4:
            return

        try:
            self.mapper = PitchMapper(
                [p.image for p in self.points],
                [p.pitch for p in self.points],
            )
        except ValueError as error:
            print(f"Homography failed: {error}")
            return

        errors = self.mapper.reprojection_errors()
        print(
            f"Homography from {len(self.points)} points | "
            f"mean error {errors.mean():.2f} m | max {errors.max():.2f} m"
        )

        if len(self.points) == 4:
            print("  (4 points fit exactly - add more points to measure accuracy)")

        for p, e in zip(self.points, errors):
            if e > 1.0:
                print(f"  WARNING: '{p.landmark}' is {e:.2f} m off - check that click.")

    def add_point(self, image_xy):
        if self.custom_pending is not None:
            name = f"custom_{self.custom_pending[0]:g}_{self.custom_pending[1]:g}"
            pitch_xy = self.custom_pending
            self.custom_pending = None
        else:
            name = self.current_name
            pitch_xy = self.landmarks[name]

        self.points = [p for p in self.points if p.landmark != name]
        self.points.append(CalibrationPoint(name, image_xy, pitch_xy))

        print(f"Placed {name}: image {image_xy} -> pitch {pitch_xy}")
        self.update_mapper()

    def remove_nearest(self, image_xy):
        if not self.points:
            return

        distances = [np.hypot(p.image[0] - image_xy[0], p.image[1] - image_xy[1]) for p in self.points]
        removed = self.points.pop(int(np.argmin(distances)))
        print(f"Removed {removed.landmark}")
        self.update_mapper()

    def undo(self):
        if self.points:
            removed = self.points.pop()
            print(f"Removed {removed.landmark}")
            self.update_mapper()

    def ask_custom(self):
        raw = input("Custom pitch point X Y in meters (empty = cancel): ").split()

        if not raw:
            return

        try:
            x, y = (float(v) for v in raw)
        except ValueError:
            print("Please enter exactly two numbers, e.g. 52.5 34")
            return

        self.custom_pending = (x, y)
        print(f"Now click the image point for ({x}, {y}).")

    # ---------- mouse ----------

    def on_mouse(self, event, x, y, flags, param):
        full = (int(round(x / self.scale)), int(round(y / self.scale)))
        self.mouse = full

        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(full)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.remove_nearest(full)

    # ---------- rendering ----------

    def render_frame(self):
        if self.show_overlay and self.mapper is not None:
            if self.overlay_cache is None:
                self.overlay_cache = draw_pitch_overlay(self.frame, self.mapper, self.dims)
            image = self.overlay_cache.copy()
        else:
            image = self.frame.copy()

        errors = self.mapper.reprojection_errors() if self.mapper is not None else None

        for i, p in enumerate(self.points):
            center = (int(p.image[0]), int(p.image[1]))
            cv2.circle(image, center, 6, (0, 255, 0), -1, cv2.LINE_AA)
            label = f"{i + 1}"
            if errors is not None and len(self.points) > 4:
                label += f" ({errors[i]:.2f}m)"
            _text(image, label, (center[0] + 8, center[1] - 8), 0.6, (0, 255, 0))

        display = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        self.draw_magnifier(image, display)
        self.draw_hud(display, errors)

        return display

    def draw_magnifier(self, image, display):
        r = MAGNIFIER_RADIUS
        mx, my = self.mouse
        padded = cv2.copyMakeBorder(image, r, r, r, r, cv2.BORDER_CONSTANT)
        crop = padded[my:my + 2 * r, mx:mx + 2 * r]

        if crop.shape[:2] != (2 * r, 2 * r):
            return

        zoom = cv2.resize(crop, None, fx=MAGNIFIER_ZOOM, fy=MAGNIFIER_ZOOM, interpolation=cv2.INTER_NEAREST)
        size = zoom.shape[0]
        c = size // 2
        cv2.line(zoom, (c, 0), (c, size), (0, 255, 255), 1)
        cv2.line(zoom, (0, c), (size, c), (0, 255, 255), 1)
        cv2.rectangle(zoom, (0, 0), (size - 1, size - 1), (255, 255, 255), 2)

        # Put the magnifier in the corner away from the cursor.
        dh, dw = display.shape[:2]
        cursor_x = mx * self.scale
        x0 = 10 if cursor_x > dw / 2 else dw - size - 10
        y0 = dh - size - 10

        if y0 > 0:
            display[y0:y0 + size, x0:x0 + size] = zoom

    def draw_hud(self, display, errors):
        if self.custom_pending is not None:
            target = f"CUSTOM {self.custom_pending} - click it"
        else:
            placed = any(p.landmark == self.current_name for p in self.points)
            target = f"[{self.current + 1}/{len(self.landmark_names)}] {self.current_name}"
            target += f" {self.landmarks[self.current_name]}" + (" (placed)" if placed else "")

        lines = [
            f"Frame {self.frame_number} | points: {len(self.points)}",
            f"Place: {target}",
        ]

        if errors is not None:
            lines.append(f"Reprojection error: mean {errors.mean():.2f} m | max {errors.max():.2f} m")

        lines.append(
            "N/P landmark | click place | right-click delete | C custom | U undo | O overlay | S save | Q quit"
        )

        y = 25
        for line in lines:
            _text(display, line, (10, y), 0.55, (255, 255, 255))
            y += 24

    def render_map(self):
        image = self.diagram.image.copy()

        placed = [p.pitch for p in self.points]
        image = self.diagram.draw_points(placed, (0, 255, 0), radius=6, image=image)

        if self.mapper is not None:
            projected = self.mapper.transform_points([p.image for p in self.points])
            image = self.diagram.draw_points(projected, (0, 0, 255), radius=3, image=image)

            # Where the cursor lands on the pitch.
            cursor = self.mapper.transform_points([self.mouse])
            image = self.diagram.draw_points(cursor, (255, 0, 255), radius=4, image=image)

        target = self.custom_pending or self.landmarks[self.current_name]
        tx, ty = self.diagram.to_px([target])[0].astype(int)
        cv2.circle(image, (tx, ty), 12, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image, (tx, ty), 3, (0, 255, 255), -1, cv2.LINE_AA)

        _text(image, "yellow = landmark to place | green = placed | red = mapped click | pink = cursor",
              (10, image.shape[0] - 10), 0.45, (255, 255, 255))

        return image

    # ---------- main loop ----------

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(MAP_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        while True:
            cv2.imshow(WINDOW, self.render_frame())
            cv2.imshow(MAP_WINDOW, self.render_map())

            key = cv2.waitKeyEx(30)

            if key == -1:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return None
                continue

            char = chr(key & 0xFF).lower() if key < 256 else ""

            if char in ("n", "]") or key in KEY_RIGHT:
                self.current = (self.current + 1) % len(self.landmark_names)
            elif char in ("p", "[") or key in KEY_LEFT:
                self.current = (self.current - 1) % len(self.landmark_names)
            elif char == "c":
                self.ask_custom()
            elif char == "u":
                self.undo()
            elif char == "o":
                self.show_overlay = not self.show_overlay
            elif char == "s":
                if len(self.points) < 4:
                    print(f"Need at least 4 points (have {len(self.points)}).")
                    continue
                return Keyframe(frame=self.frame_number, points=self.points)
            elif char == "q" or key == 27:
                return None


def _text(image, text, origin, scale, color):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    args = parse_args()
    dims = PitchDimensions(length=args.length, width=args.width)
    output = Path(args.output)

    calibration = Calibration(video_path=args.video, pitch=dims)

    if output.exists():
        try:
            calibration = Calibration.load(output)
            print(f"Loaded {output} ({len(calibration.keyframes)} keyframes)")
        except (ValueError, KeyError) as error:
            print(f"Ignoring existing {output}: {error}")

    existing = calibration.get_keyframe(args.frame)
    existing_points = existing.points if existing else []

    frame = read_frame(args.video, args.frame)
    print(__doc__)

    keyframe = CalibrationTool(frame, args.frame, calibration.pitch, existing_points).run()
    cv2.destroyAllWindows()

    if keyframe is None:
        print("Calibration cancelled. Nothing saved.")
        return

    calibration.set_keyframe(keyframe)
    calibration.save(output)

    errors = keyframe.build_mapper().reprojection_errors()
    print()
    print("==============================")
    print("Calibration saved!")
    print("==============================")
    print(f"File: {output}")
    print(f"Keyframes: {[k.frame for k in calibration.keyframes]}")
    print(f"Frame {keyframe.frame}: {len(keyframe.points)} points, mean error {errors.mean():.2f} m")


if __name__ == "__main__":
    main()
