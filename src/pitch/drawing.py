import cv2
import numpy as np

from src.pitch.landmarks import PitchDimensions, get_pitch_lines


GRASS = (60, 120, 50)
WHITE = (255, 255, 255)


def project_pitch_to_image(matrix_pitch_to_image, points):
    """Project pitch points into the image, returning NaN for points behind the camera."""

    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix_pitch_to_image.T

    w = homogeneous[:, 2:3]

    # A homography is only defined up to scale (including sign): points in
    # front of the camera are the ones sharing the majority sign of w.
    if np.median(w) < 0:
        w = -w
        homogeneous = -homogeneous

    projected = homogeneous[:, :2] / np.where(np.abs(w) < 1e-12, np.nan, w)
    projected[w.ravel() <= 0] = np.nan

    return projected


def get_dense_pitch_lines(dims=PitchDimensions(), step=0.5):
    """Pitch lines with straight segments subdivided, so perspective is followed correctly."""

    return [densify(line, step) for line in get_pitch_lines(dims)]


def draw_pitch_lines(image, matrix_pitch_to_image, dense_lines, color, thickness, line_type=cv2.LINE_AA):
    """Draw pitch lines (in meters) onto an image, in place."""

    for line in dense_lines:
        projected = project_pitch_to_image(matrix_pitch_to_image, line)

        valid = ~np.isnan(projected).any(axis=1)
        projected = np.round(np.clip(np.nan_to_num(projected), -1e5, 1e5)).astype(np.int32)

        for i in range(1, len(projected)):
            if valid[i - 1] and valid[i]:
                cv2.line(image, tuple(projected[i - 1]), tuple(projected[i]), color, thickness, line_type)

    return image


def draw_pitch_overlay(frame, mapper, dims=PitchDimensions(), color=(0, 0, 255), thickness=2):
    """Draw the theoretical pitch lines on the frame using the calibration.

    If the calibration is good, the drawn lines sit exactly on the real ones.
    """

    return draw_pitch_lines(
        frame.copy(), mapper.inverse_matrix, get_dense_pitch_lines(dims), color, thickness
    )


class PitchDiagram:
    """Top-down 2D pitch image with meter <-> pixel conversion."""

    def __init__(self, dims=PitchDimensions(), scale=8, margin=5.0):
        self.dims = dims
        self.scale = scale
        self.margin = margin

        width = int((dims.length + 2 * margin) * scale)
        height = int((dims.width + 2 * margin) * scale)

        self.image = np.full((height, width, 3), GRASS, dtype=np.uint8)

        for line in get_pitch_lines(dims):
            cv2.polylines(
                self.image,
                [self.to_px(line).astype(np.int32)],
                False,
                WHITE,
                2,
                cv2.LINE_AA,
            )

    def to_px(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return (points + self.margin) * self.scale

    def draw_points(self, points, color, radius=5, labels=None, image=None):
        image = self.image.copy() if image is None else image

        for i, (x, y) in enumerate(self.to_px(points)):
            if not np.isfinite([x, y]).all():
                continue

            center = (int(round(x)), int(round(y)))
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)

            if labels is not None:
                cv2.putText(
                    image, str(labels[i]), (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )

        return image


def densify(polyline, step):
    points = [polyline[0]]

    for a, b in zip(polyline[:-1], polyline[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            points.append(a + t * (b - a))

    return np.array(points)
