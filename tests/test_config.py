import pytest

from src.utils.config import Config, load_config


def test_default_config_file_loads():
    config = load_config("configs/default.yaml")

    assert config.pitch.length == 105.0
    assert config.paths.tracks.endswith("tracks.csv")


def test_missing_keys_use_defaults(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("video: other.mp4\npitch:\n  length: 100\n")

    config = load_config(path)

    assert config.video == "other.mp4"
    assert config.pitch.length == 100
    assert config.pitch.width == Config().pitch.width


def test_unknown_section_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("pitchh:\n  length: 100\n")

    with pytest.raises(ValueError, match="pitchh"):
        load_config(path)
