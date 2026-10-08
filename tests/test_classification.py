import numpy as np
import pytest

from src.classification.team_classifier import OTHER, TeamClassifier, jersey_color


def lab(bgr):
    import cv2
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def test_two_teams_and_referee():
    rng = np.random.default_rng(1)
    red, navy, cyan = lab((40, 40, 210)), lab((70, 30, 20)), lab((220, 220, 60))

    colors = {}
    for i in range(10):
        colors[i] = red + rng.normal(0, 2, 3)
        colors[100 + i] = navy + rng.normal(0, 2, 3)
    colors[999] = cyan

    teams = TeamClassifier().fit_predict(colors)

    assert len({teams[i] for i in range(10)}) == 1
    assert len({teams[100 + i] for i in range(10)}) == 1
    assert teams[0] != teams[100]
    assert teams[999] == OTHER


def test_jersey_color_ignores_grass():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:] = (40, 140, 40)                 # grass
    frame[60:100, 85:115] = (40, 40, 210)    # red torso, narrower than the box

    color = jersey_color(frame, (70, 40, 130, 160))

    assert color == pytest.approx(lab((40, 40, 210)), abs=2)


def test_jersey_color_rejects_tiny_boxes():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert jersey_color(frame, (10, 10, 12, 14)) is None
