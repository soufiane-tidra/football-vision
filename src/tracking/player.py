from collections import Counter
from dataclasses import dataclass, field


@dataclass
class PlayerTrack:
    track_id: int
    frames: list[int] = field(default_factory=list)

    # Image position in pixels: bottom-center of the box (where the feet touch the pitch).
    positions: list[tuple[float, float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    # Pitch position in meters, filled by src.pitch.projection (NaN = off the pitch).
    pitch_positions: list[tuple[float, float]] = field(default_factory=list)

    # How often the detector assigned each class to this track.
    class_counts: Counter = field(default_factory=Counter)

    def add_detection(
        self,
        frame: int,
        x: float,
        y: float,
        confidence: float,
        class_name: str | None = None
    ):
        self.frames.append(frame)
        self.positions.append((x, y))
        self.confidences.append(confidence)

        if class_name is not None:
            self.class_counts[class_name] += 1

    @property
    def role(self):
        """Most frequent class of the track (single-frame misclassifications are outvoted)."""

        if not self.class_counts:
            return None

        return self.class_counts.most_common(1)[0][0]

    @property
    def detection_count(self):
        return len(self.frames)

    @property
    def first_frame(self):
        return self.frames[0] if self.frames else None

    @property
    def last_frame(self):
        return self.frames[-1] if self.frames else None
