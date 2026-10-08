import numpy as np
import pytest

from src.pitch.calibration import Calibration, CalibrationPoint, Keyframe
from src.pitch.camera_motion import _blend
from src.pitch.drawing import project_pitch_to_image
from src.pitch.homography import PitchMapper
from src.pitch.landmarks import PitchDimensions, get_landmarks, get_pitch_lines
from src.pitch.projection import on_pitch_ratio, project_tracks_to_pitch
from src.tracking.player import PlayerTrack


# A plausible broadcast-like homography: pitch meters -> image pixels.
PITCH_TO_IMAGE = np.array([
    [14.0, -4.0, 600.0],
    [0.3, 6.0, 300.0],
    [0.0005, 0.004, 1.0],
])
IMAGE_TO_PITCH = np.linalg.inv(PITCH_TO_IMAGE)


def to_image(points):
    return project_pitch_to_image(PITCH_TO_IMAGE, points)


# ---------- landmarks ----------

def test_landmarks_are_symmetric():
    dims = PitchDimensions()
    landmarks = get_landmarks(dims)

    for name, (x, y) in landmarks.items():
        if name.startswith("left_"):
            mirror_x, mirror_y = landmarks["right_" + name[len("left_"):]]
            assert mirror_x == pytest.approx(dims.length - x)
            assert mirror_y == pytest.approx(y)


def test_penalty_arc_points_lie_on_arc():
    dims = PitchDimensions()
    landmarks = get_landmarks(dims)
    spot = np.array(landmarks["left_penalty_spot"])

    for name in ("left_penalty_arc_top", "left_penalty_arc_bottom"):
        distance = np.linalg.norm(np.array(landmarks[name]) - spot)
        assert distance == pytest.approx(dims.center_circle_radius)


def test_landmarks_follow_pitch_dimensions():
    landmarks = get_landmarks(PitchDimensions(length=100, width=64))

    assert landmarks["center_spot"] == (50, 32)
    assert landmarks["corner_bottom_right"] == (100, 64)


def test_pitch_lines_stay_inside_pitch():
    dims = PitchDimensions()

    for line in get_pitch_lines(dims):
        assert line.min() >= -1e-9
        assert line[:, 0].max() <= dims.length + 1e-9
        assert line[:, 1].max() <= dims.width + 1e-9


# ---------- homography ----------

def make_correspondences(names):
    landmarks = get_landmarks()
    pitch = np.array([landmarks[n] for n in names])
    return to_image(pitch), pitch


NAMES = [
    "corner_top_left", "left_penalty_corner_top", "left_penalty_corner_bottom",
    "left_goal_area_corner_top", "left_penalty_spot", "halfway_top",
    "center_spot", "halfway_bottom",
]


def test_mapper_recovers_known_homography():
    image, pitch = make_correspondences(NAMES)
    mapper = PitchMapper(image, pitch)

    assert mapper.reprojection_errors().max() < 1e-3

    x, y = mapper.transform_point(*to_image([[30.0, 20.0]])[0])
    assert (x, y) == pytest.approx((30.0, 20.0), abs=1e-3)


def test_inverse_transform_roundtrip():
    image, pitch = make_correspondences(NAMES)
    mapper = PitchMapper(image, pitch)

    back = mapper.inverse_transform_points(mapper.transform_points(image))
    np.testing.assert_allclose(back, image, atol=1e-3)


def test_ransac_rejects_a_wrong_click():
    image, pitch = make_correspondences(NAMES)
    image[2] += [60, -40]           # one badly clicked point

    mapper = PitchMapper(image, pitch)

    assert not mapper.inlier_mask[2]
    assert mapper.inlier_mask.sum() == len(NAMES) - 1


def test_mapper_needs_four_points():
    image, pitch = make_correspondences(NAMES[:3])

    with pytest.raises(ValueError):
        PitchMapper(image, pitch)


def test_projection_independent_of_homography_scale_and_sign():
    points = np.array([[10.0, 10.0], [52.5, 34.0], [100.0, 60.0]])

    expected = project_pitch_to_image(PITCH_TO_IMAGE, points)
    flipped = project_pitch_to_image(-3.0 * PITCH_TO_IMAGE, points)

    np.testing.assert_allclose(flipped, expected)
    assert not np.isnan(expected).any()


def test_blend_handles_opposite_signs():
    blended = _blend(PITCH_TO_IMAGE, -2.0 * PITCH_TO_IMAGE, 0.5)

    np.testing.assert_allclose(
        blended / blended[2, 2], PITCH_TO_IMAGE / PITCH_TO_IMAGE[2, 2], atol=1e-9
    )


# ---------- calibration file ----------

def test_calibration_roundtrip(tmp_path):
    image, pitch = make_correspondences(NAMES)
    points = [CalibrationPoint(n, tuple(i), tuple(p)) for n, i, p in zip(NAMES, image, pitch)]

    calibration = Calibration(video_path="match.mp4")
    calibration.set_keyframe(Keyframe(120, points))
    calibration.set_keyframe(Keyframe(0, points[:5]))

    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = Calibration.load(path)

    assert [k.frame for k in loaded.keyframes] == [0, 120]
    assert loaded.get_keyframe(120).points[0].landmark == NAMES[0]
    np.testing.assert_allclose(
        loaded.get_keyframe(120).build_mapper().matrix,
        calibration.get_keyframe(120).build_mapper().matrix,
        atol=1e-6,
    )


def test_old_calibration_format_is_rejected(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"image_points": [], "pitch_points": []}')

    with pytest.raises(ValueError, match="old calibration format"):
        Calibration.load(path)


# ---------- projection ----------

def test_project_tracks_marks_off_pitch_positions():
    homographies = np.array([IMAGE_TO_PITCH] * 3)

    player = PlayerTrack(track_id=7)
    on_pitch = to_image([[40.0, 30.0]])[0]
    off_pitch = to_image([[40.0, -20.0]])[0]      # 20 m behind the touchline

    player.add_detection(0, *on_pitch, 0.9)
    player.add_detection(1, *off_pitch, 0.9)
    player.add_detection(5, *on_pitch, 0.9)        # frame without homography

    project_tracks_to_pitch({7: player}, homographies)

    assert player.pitch_positions[0] == pytest.approx((40.0, 30.0), abs=1e-3)
    assert np.isnan(player.pitch_positions[1]).all()
    assert np.isnan(player.pitch_positions[2]).all()
    assert on_pitch_ratio(player) == pytest.approx(1 / 3)
