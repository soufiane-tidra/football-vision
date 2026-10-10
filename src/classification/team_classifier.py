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


def collect_track_colors(video_path, tracks, every_n_frames=5, min_samples=2, max_frames=None):
    """Median jersey color (Lab) per track, sampled every few frames of the video.

    tracks: track_id -> PlayerTrack with boxes. Returns track_id -> color.
    """

    by_frame = {}
    for track_id, track in tracks.items():
        for frame, box in zip(track.frames, track.boxes):
            if frame % every_n_frames == 0:
                by_frame.setdefault(frame, []).append((track_id, box))

    samples = {}
    video = cv2.VideoCapture(str(video_path))
    frame_number = 0

    while by_frame and (max_frames is None or frame_number < max_frames):
        success, frame = video.read()
        if not success:
            break

        for track_id, box in by_frame.pop(frame_number, []):
            color = jersey_color(frame, box)
            if color is not None:
                samples.setdefault(track_id, []).append(color)

        frame_number += 1

    video.release()

    return {t: np.median(c, axis=0) for t, c in samples.items() if len(c) >= min_samples}


class TeamClassifier:
    """Cluster jersey colors into teams, leaving other kits (referees, goalkeepers) out.

    Colors are clustered into n_clusters groups (one more than the number of
    teams, to give referee / goalkeeper kits somewhere to go). The n_teams
    biggest groups are the teams. A leftover group whose color is close to a
    team is merged into it (it is just that team under different lighting).

    Brightness (L) is down-weighted: fog, shadows and floodlights change it
    much more than they change the hue.
    """

    def __init__(self, n_teams=2, n_clusters=3, luminance_weight=0.5, merge_distance=15.0, outlier_factor=2.5):
        self.n_teams = n_teams
        self.n_clusters = max(n_clusters, n_teams)
        self.luminance_weight = luminance_weight
        self.merge_distance = merge_distance
        self.outlier_factor = outlier_factor

        self.centers = None          # feature space, team clusters first
        self.cluster_team = None     # cluster index -> team (or OTHER)
        self.threshold = None

    def _features(self, colors):
        features = np.asarray(colors, dtype=np.float32).reshape(-1, 3).copy()
        features[:, 0] *= self.luminance_weight
        return features

    def fit_predict(self, track_colors: dict[int, np.ndarray]) -> dict[int, int]:
        """track_colors: track_id -> Lab color. Returns track_id -> team (0, 1 or OTHER)."""

        ids = list(track_colors)
        data = self._features([track_colors[i] for i in ids])

        if len(data) < self.n_teams:
            raise ValueError("Not enough tracks to classify teams.")

        k = min(self.n_clusters, len(data))
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

        # Biggest clusters first: those are the teams.
        order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")
        centers = centers[order]
        labels = np.argsort(order)[labels]

        cluster_team = list(range(self.n_teams)) + [OTHER] * (k - self.n_teams)
        for cluster in range(self.n_teams, k):
            distances = np.linalg.norm(centers[: self.n_teams] - centers[cluster], axis=1)
            if distances.min() < self.merge_distance:
                cluster_team[cluster] = int(distances.argmin())

        self.centers = centers
        self.cluster_team = cluster_team

        distances = np.linalg.norm(data - centers[labels], axis=1)
        # Never tighter than merge_distance: very clean clusters would otherwise
        # reject an ordinary lighting change.
        self.threshold = max(self.outlier_factor * float(np.median(distances)), self.merge_distance)

        return {track_id: self.predict(track_colors[track_id]) for track_id in ids}

    def predict(self, color) -> int:
        """Team of one jersey color (0, 1 or OTHER), using the fitted clusters."""

        distances = np.linalg.norm(self.centers - self._features(color), axis=1)
        cluster = int(distances.argmin())

        if distances[cluster] > self.threshold:
            return OTHER

        return self.cluster_team[cluster]

    def team_display_colors(self):
        """Bright BGR drawing color for each team, derived from its jersey color."""

        colors = []

        for center in self.centers[: self.n_teams]:
            lab = np.uint8([[[center[0] / self.luminance_weight, center[1], center[2]]]])
            hsv = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV)[0, 0]
            hsv = np.uint8([[[hsv[0], max(int(hsv[1]), 170), 255]]])
            colors.append(tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))

        return colors
