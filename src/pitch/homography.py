import cv2
import numpy as np


class PitchMapper:
    """Maps image pixels <-> pitch meters with a planar homography."""

    def __init__(self, image_points, pitch_points, ransac_threshold_m=1.0):

        if len(image_points) < 4:
            raise ValueError(
                "At least 4 point correspondences are required."
            )

        if len(image_points) != len(pitch_points):
            raise ValueError(
                "image_points and pitch_points must have the same length."
            )

        self.image_points = np.array(image_points, dtype=np.float32)
        self.pitch_points = np.array(pitch_points, dtype=np.float32)

        # RANSAC needs redundancy; with exactly 4 points use a direct solve.
        method = cv2.RANSAC if len(image_points) > 4 else 0

        # The threshold is in destination units, i.e. meters.
        matrix, mask = cv2.findHomography(
            self.image_points,
            self.pitch_points,
            method,
            ransac_threshold_m
        )

        if matrix is None:
            raise ValueError(
                "Could not calculate homography. "
                "Check that points are not collinear."
            )

        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(matrix)
        self.inlier_mask = mask.ravel().astype(bool)

    @classmethod
    def from_matrix(cls, matrix):
        mapper = cls.__new__(cls)
        mapper.matrix = np.asarray(matrix, dtype=np.float64)
        mapper.inverse_matrix = np.linalg.inv(mapper.matrix)
        mapper.image_points = np.empty((0, 2), dtype=np.float32)
        mapper.pitch_points = np.empty((0, 2), dtype=np.float32)
        mapper.inlier_mask = np.empty(0, dtype=bool)
        return mapper

    def transform_point(self, x, y):

        return tuple(self.transform_points([(x, y)])[0])

    def transform_points(self, points):
        """Image pixels -> pitch meters. Returns an (N, 2) array."""

        return _apply(self.matrix, points)

    def inverse_transform_points(self, points):
        """Pitch meters -> image pixels. Returns an (N, 2) array."""

        return _apply(self.inverse_matrix, points)

    def reprojection_errors(self):
        """Distance (meters) between each mapped image point and its true pitch point."""

        projected = self.transform_points(self.image_points)

        return np.linalg.norm(projected - self.pitch_points, axis=1)


def _apply(matrix, points):

    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)

    if len(points) == 0:
        return np.empty((0, 2))

    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
