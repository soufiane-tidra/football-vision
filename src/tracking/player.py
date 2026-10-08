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

    def add_detection(
        self,
        frame: int,
        x: float,
        y: float,
        confidence: float
    ):
        self.frames.append(frame)
        self.positions.append((x, y))
        self.confidences.append(confidence)

    @property
    def detection_count(self):
        return len(self.frames)

    @property
    def first_frame(self):
        return self.frames[0] if self.frames else None

    @property
    def last_frame(self):
        return self.frames[-1] if self.frames else None
