from src.utils.config import load_config
from src.video.frames import extract_frames


config = load_config()

saved = extract_frames(
    config.video,
    config.paths.frames,
    every_n_frames=config.frames.every_n_frames
)

print(f"Saved frames: {saved}")
print(f"Folder: {config.paths.frames}")
