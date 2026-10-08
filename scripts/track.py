from src.detection.detector import FootballDetector
from src.utils.config import load_config


config = load_config()

detector = FootballDetector(model_path=config.detection.model)

csv_path = detector.track_video(
    config.video,
    csv_path=config.paths.tracks,
    tracker=config.detection.tracker
)

print("Tracking completed.")
print(f"Tracking data saved to: {csv_path}")
