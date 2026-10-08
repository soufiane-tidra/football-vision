from src.utils.config import load_config
from src.video.video import get_video_info


config = load_config()

info = get_video_info(config.video)

print("Video Information")
print("-----------------")
print(f"File: {config.video}")
print(f"FPS: {info['fps']}")
print(f"Resolution: {info['width']}x{info['height']}")
print(f"Frames: {info['frame_count']}")
print(f"Duration: {info['duration']:.2f} seconds")
