import numpy as np
import pytest

from src.pitch.drawing import project_pitch_to_image
from src.pitch.keypoints import (
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    homography_from_keypoints,
    pitch_keypoints,
    smooth_homographies,
)
from src.pitch.landmarks import PitchDimensions, get_landmarks


PITCH_TO_IMAGE = np.array([
    [14.0, -4.0, 600.0],
    [0.3, 6.0, 300.0],
    [0.0005, 0.004, 1.0],
])


def test_keypoint_layout_matches_landmarks():
    points = pitch_keypoints()
    landmarks = get_landmarks()

    assert points.shape == (NUM_KEYPOINTS, 2) == (32, 2)
    assert len(set(KEYPOINT_NAMES)) == 32

    for name, point in zip(KEYPOINT_NAMES, points):
        if name in landmarks:
            assert tuple(point) == pytest.approx(landmarks[name]), name


def test_keypoints_are_mirror_symmetric():
    dims = PitchDimensions()
    points = {n: p for n, p in zip(KEYPOINT_NAMES, pitch_keypoints(dims))}

    for name, (x, y) in points.items():
        if name.startswith("left_"):
            mx, my = points["right_" + name[len("left_"):]]
            assert (mx, my) == pytest.approx((dims.length - x, y))


def detections(indices, noise=0.0, seed=0):
    pitch = pitch_keypoints()
    image = project_pitch_to_image(PITCH_TO_IMAGE, pitch)
    image += np.random.default_rng(seed).normal(0, noise, image.shape)

    confidence = np.zeros(NUM_KEYPOINTS)
    confidence[indices] = 0.9
    return image, confidence, pitch


LEFT_HALF = [0, 1, 6, 7, 8, 9, 12, 13, 14, 15, 16]


def test_homography_from_good_detections():
    image, confidence, pitch = detections(LEFT_HALF, noise=0.5)
    mapper = homography_from_keypoints(image, confidence, pitch)

    assert mapper is not None
    x, y = mapper.transform_point(*project_pitch_to_image(PITCH_TO_IMAGE, [[30.0, 40.0]])[0])
    assert (x, y) == pytest.approx((30.0, 40.0), abs=0.3)


def test_low_confidence_points_are_ignored():
    image, confidence, pitch = detections(LEFT_HALF)
    image[0] += 300                       # garbage location...
    confidence[0] = 0.1                   # ...but low confidence

    mapper = homography_from_keypoints(image, confidence, pitch)

    assert mapper is not None
    assert mapper.reprojection_errors().max() < 0.05


def test_too_few_points_rejected():
    image, confidence, pitch = detections([0, 1, 6])
    assert homography_from_keypoints(image, confidence, pitch) is None


def test_collinear_points_rejected():
    # Only the halfway line is visible: 4 points on one line.
    image, confidence, pitch = detections([13, 14, 15, 16])
    assert homography_from_keypoints(image, confidence, pitch) is None


def test_inconsistent_detections_rejected():
    image, confidence, pitch = detections([0, 1, 6, 7, 8])
    image[[1, 3, 6]] += [[80, 0], [0, 90], [-70, 40]]   # 3 of 5 points wrong

    assert homography_from_keypoints(image, confidence, pitch) is None


def test_smoothing_removes_jitter_and_keeps_scale_invariance():
    rng = np.random.default_rng(3)
    base = np.linalg.inv(PITCH_TO_IMAGE)

    noisy = np.array([
        base * (1 + rng.normal(0, 0.01, (3, 3))) * (-1 if i % 2 else 2.0)   # random sign/scale
        for i in range(21)
    ])
    valid = np.ones(21, dtype=bool)

    smoothed = smooth_homographies(noisy, valid, window=9)

    def error(H):
        point = project_pitch_to_image(np.linalg.inv(H), [[52.5, 34.0]])[0]
        return np.linalg.norm(point - project_pitch_to_image(PITCH_TO_IMAGE, [[52.5, 34.0]])[0])

    assert np.mean([error(h) for h in smoothed[5:-5]]) < np.mean([error(h) for h in noisy[5:-5]])


def test_smoothing_skips_invalid_frames():
    H = np.array([np.eye(3), np.eye(3) * 5, np.eye(3)])
    smoothed = smooth_homographies(H, np.array([True, False, True]), window=3)

    np.testing.assert_allclose(smoothed[1] / smoothed[1][2, 2], np.eye(3))
