"""Project configuration loaded from configs/*.yaml."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("configs/default.yaml")


@dataclass(frozen=True)
class Paths:
    frames: str = "data/processed/frames"
    tracks: str = "data/processed/tracks.csv"
    calibration: str = "data/processed/pitch_calibration.json"
    homographies: str = "data/processed/homographies.npy"
    tracks_pitch: str = "data/processed/tracks_pitch.csv"
    player_metrics: str = "data/processed/player_metrics.csv"
    players: str = "data/processed/players.csv"
    ball: str = "data/processed/ball.csv"
    possession: str = "data/processed/possession.csv"
    events: str = "data/processed/events.csv"
    team_shape: str = "data/processed/team_shape.csv"
    outputs: str = "outputs"


@dataclass(frozen=True)
class DetectionConfig:
    model: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    conf: float = 0.25


@dataclass(frozen=True)
class FramesConfig:
    every_n_frames: int = 30


@dataclass(frozen=True)
class PitchConfig:
    length: float = 105.0
    width: float = 68.0


@dataclass(frozen=True)
class MovementConfig:
    smoothing_s: float = 0.4
    max_speed_mps: float = 12.0
    min_time_on_pitch_s: float = 1.0
    min_on_pitch_ratio: float = 0.5


@dataclass(frozen=True)
class RenderConfig:
    gif_start_s: float = 20.0
    gif_duration_s: float = 10.0
    gif_width: int = 640
    gif_fps: float = 7.5


@dataclass(frozen=True)
class PitchKeypointsConfig:
    workspace: str = "roboflow-jvuqo"
    project: str = "football-field-detection-f07vi"
    version: int = 12
    dataset_dir: str = "data/datasets/pitch_keypoints"
    base_model: str = "yolo11s-pose.pt"
    weights: str = "models/pitch_keypoints.pt"
    epochs: int = 300
    imgsz: int = 640
    batch: int = 16
    confidence: float = 0.5
    min_points: int = 6
    max_error_m: float = 1.0
    smoothing_frames: int = 31


@dataclass(frozen=True)
class PlayerDetectionConfig:
    workspace: str = "roboflow-jvuqo"
    project: str = "football-players-detection-3zvbc"
    version: int = 12
    dataset_dir: str = "data/datasets/players"
    base_model: str = "yolo11s.pt"
    weights: str = "models/football_detector.pt"
    epochs: int = 100
    imgsz: int = 1280
    batch: int = 6


@dataclass(frozen=True)
class StitchingConfig:
    max_gap_s: float = 3.0
    max_speed_mps: float = 8.0
    slack_m: float = 3.0
    min_player_s: float = 1.0
    touchline_band_m: float = 1.5


@dataclass(frozen=True)
class BallConfig:
    confidence: float = 0.05
    max_speed_px: float = 70.0
    max_gap_frames: int = 30
    static_s: float = 4.0
    min_segment_confidence: float = 0.2
    min_travel_m: float = 10.0


@dataclass(frozen=True)
class Config:
    video: str = "data/raw/match.mp4"
    paths: Paths = field(default_factory=Paths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    pitch_keypoints: PitchKeypointsConfig = field(default_factory=PitchKeypointsConfig)
    player_detection: PlayerDetectionConfig = field(default_factory=PlayerDetectionConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    ball: BallConfig = field(default_factory=BallConfig)


_SECTIONS = {
    "paths": Paths,
    "detection": DetectionConfig,
    "frames": FramesConfig,
    "pitch": PitchConfig,
    "movement": MovementConfig,
    "render": RenderConfig,
    "pitch_keypoints": PitchKeypointsConfig,
    "player_detection": PlayerDetectionConfig,
    "stitching": StitchingConfig,
    "ball": BallConfig,
}


def load_config(path=DEFAULT_CONFIG) -> Config:
    """Load a YAML config; missing keys fall back to the defaults above."""

    path = Path(path)

    if not path.exists():
        return Config()

    with open(path, "r") as file:
        raw = yaml.safe_load(file) or {}

    unknown = set(raw) - set(_SECTIONS) - {"video"}
    if unknown:
        raise ValueError(f"Unknown config sections in {path}: {sorted(unknown)}")

    sections = {name: cls(**(raw.get(name) or {})) for name, cls in _SECTIONS.items()}

    return Config(video=raw.get("video", Config.video), **sections)
