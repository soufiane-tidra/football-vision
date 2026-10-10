import pytest

from src.analytics.team_shape import defending_left, shape_of, summarize, team_shapes


RECTANGLE = [(20.0, 10.0), (20.0, 50.0), (40.0, 10.0), (40.0, 50.0), (30.0, 30.0)]


def test_shape_of_a_rectangle():
    shape = shape_of(RECTANGLE, frame=7, defends_left=True)

    assert shape.frame == 7 and shape.players == 5
    assert (shape.centroid_x, shape.centroid_y) == pytest.approx((30.0, 30.0))
    assert shape.width == pytest.approx(40.0)          # across the pitch
    assert shape.depth == pytest.approx(20.0)          # along the pitch
    assert shape.area == pytest.approx(800.0)          # the inner player does not change the hull


def test_defensive_line_is_the_second_deepest_player():
    positions = [(8.0, 30.0), (22.0, 20.0), (24.0, 40.0), (50.0, 34.0)]

    assert shape_of(positions, defends_left=True).line_height == pytest.approx(22.0)
    assert shape_of(positions, defends_left=False).line_height == pytest.approx(105.0 - 24.0)


def test_compact_team_has_smaller_area_than_stretched_team():
    compact = [(30.0 + dx, 30.0 + dy) for dx in (0, 5, 10) for dy in (0, 5, 10)]
    stretched = [(20.0 + 3 * dx, 10.0 + 4 * dy) for dx in (0, 5, 10) for dy in (0, 5, 10)]

    assert shape_of(compact).area < shape_of(stretched).area


def test_frames_with_too_few_visible_players_are_skipped():
    frames = {0: RECTANGLE[:3], 1: RECTANGLE + [(25.0, 20.0), (35.0, 40.0)], 2: []}
    shapes = team_shapes(frames, min_players=7)

    assert [s.frame for s in shapes] == [1]


def test_defending_side_from_deepest_players():
    left = [(5.0 + i, 30.0) for i in range(40)]        # reaches x = 5
    right = [(50.0 + i, 30.0) for i in range(40)]

    assert defending_left({"team_1": right, "team_2": left}) == "team_2"


def test_summary_is_the_mean_over_frames():
    shapes = [shape_of(RECTANGLE, 0), shape_of([(x + 10, y) for x, y in RECTANGLE], 1)]
    summary = summarize(shapes)

    assert summary["width"] == pytest.approx(40.0)
    assert summary["centroid_x"] == pytest.approx(35.0)
    assert summarize([]) == {}


def test_report_charts_render_to_png():
    from src.visualization.charts import passing_network_chart, team_shape_chart

    colors = {"team_1": (230, 150, 50), "team_2": (60, 60, 230)}
    nodes = {1: (30.0, 30.0, "team_1"), 2: (45.0, 40.0, "team_1")}
    shapes = {"team_1": [shape_of(RECTANGLE, frame) for frame in range(0, 90, 10)], "team_2": []}

    for png in (passing_network_chart(nodes, {(1, 2): 2}, colors), team_shape_chart(shapes, colors, fps=30.0)):
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000
