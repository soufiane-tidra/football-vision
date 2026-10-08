"""Unsupervised team classification from jersey colors.

For each track, the torso region of several detections is sampled, grass
pixels are removed, and the mean color (Lab space) is computed. Tracks are
then clustered into two teams with K-means. Tracks far from both team colors
(referees, goalkeepers) are labelled OTHER.
"""

import cv2
import numpy as np


OTHER = -1


def jersey_color(frame, box):
    """Mean Lab color of the torso inside a bounding box, ignoring grass. None if unusable."""

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h = y2 - y1

    # Torso: below the head, above the shorts.
    top = y1 + int(0.15 * h)
    bottom = y1 + int(0.50 * h)
    left = x1 + int(0.2 * (x2 - x1))
    right = x2 - int(0.2 * (x2 - x1))

    crop = frame[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 3:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[~grass]

    if len(pixels) < 10:
        return None

    return pixels.astype(np.float32).mean(axis=0)


class TeamClassifier:

    def __init__(self, n_teams=2, outlier_factor=2.5):
        self.n_teams = n_teams
        self.outlier_factor = outlier_factor
        self.centers = None
        self.threshold = None

    def fit_predict(self, track_colors: dict[int, np.ndarray]) -> dict[int, int]:
        """track_colors: track_id -> Lab color. Returns track_id -> team (0, 1 or OTHER)."""

        ids = list(track_colors)
        data = np.array([track_colors[i] for i in ids], dtype=np.float32)

        if len(data) < self.n_teams:
            raise ValueError("Not enough tracks to classify teams.")

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(
            data, self.n_teams, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        labels = labels.ravel()
        distances = np.linalg.norm(data - centers[labels], axis=1)

        self.centers = centers
        self.threshold = self.outlier_factor * max(float(np.median(distances)), 1.0)

        return {
            track_id: int(label) if distance <= self.threshold else OTHER
            for track_id, label, distance in zip(ids, labels, distances)
        }

    def team_display_colors(self):
        """Bright BGR drawing color for each team, derived from its jersey color."""

        colors = []

        for center in self.centers:
            lab = np.uint8([[center]])
            hsv = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV)[0, 0]
            hsv = np.uint8([[[hsv[0], max(int(hsv[1]), 170), 255]]])
            colors.append(tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))

        return colors
