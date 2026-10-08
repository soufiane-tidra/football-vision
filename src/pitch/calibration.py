"""Load and save pitch calibration files.

A calibration contains one or more keyframes. Each keyframe holds manually
clicked image points and their pitch coordinates for that video frame:

{
    "video_path": "data/raw/match.mp4",
    "pitch": {"length": 105.0, "width": 68.0},
    "keyframes": [
        {
            "frame": 0,
            "points": [
                {"landmark": "halfway_top", "image": [1652, 341], "pitch": [52.5, 0.0]},
                ...
            ]
        }
    ]
}
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.pitch.homography import PitchMapper
from src.pitch.landmarks import PitchDimensions


@dataclass
class CalibrationPoint:
    landmark: str
    image: tuple[float, float]
    pitch: tuple[float, float]


@dataclass
class Keyframe:
    frame: int
    points: list[CalibrationPoint] = field(default_factory=list)

    def build_mapper(self) -> PitchMapper:
        return PitchMapper(
            [p.image for p in self.points],
            [p.pitch for p in self.points],
        )


@dataclass
class Calibration:
    video_path: str
    pitch: PitchDimensions = field(default_factory=PitchDimensions)
    keyframes: list[Keyframe] = field(default_factory=list)

    def get_keyframe(self, frame: int) -> Keyframe | None:
        for keyframe in self.keyframes:
            if keyframe.frame == frame:
                return keyframe
        return None

    def set_keyframe(self, keyframe: Keyframe):
        self.keyframes = [k for k in self.keyframes if k.frame != keyframe.frame]
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.frame)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "video_path": self.video_path,
            "pitch": {"length": self.pitch.length, "width": self.pitch.width},
            "keyframes": [
                {
                    "frame": k.frame,
                    "points": [
                        {
                            "landmark": p.landmark,
                            "image": [float(v) for v in p.image],
                            "pitch": [float(v) for v in p.pitch],
                        }
                        for p in k.points
                    ],
                }
                for k in self.keyframes
            ],
        }

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load(cls, path) -> "Calibration":
        with open(path, "r") as file:
            data = json.load(file)

        if "keyframes" not in data:
            raise ValueError(
                f"{path} uses the old calibration format. "
                "Re-run: python -m scripts.calibrate_pitch"
            )

        pitch = PitchDimensions(**data.get("pitch", {}))

        keyframes = [
            Keyframe(
                frame=int(k["frame"]),
                points=[
                    CalibrationPoint(
                        landmark=p["landmark"],
                        image=tuple(p["image"]),
                        pitch=tuple(p["pitch"]),
                    )
                    for p in k["points"]
                ],
            )
            for k in data["keyframes"]
        ]

        return cls(video_path=data["video_path"], pitch=pitch, keyframes=keyframes)
