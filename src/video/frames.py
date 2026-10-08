from pathlib import Path

import cv2


def extract_frames(video_path, output_dir, every_n_frames=30):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_frames = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_index % every_n_frames == 0:
            output_path = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_frames += 1

        frame_index += 1

    video.release()

    return saved_frames

def read_frame(video_path, frame_number):
    """Return one frame of a video (BGR image)."""

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Sequential reading is exact; seeking can land on the wrong frame.
    frame = None
    for _ in range(frame_number + 1):
        success, frame = video.read()
        if not success:
            video.release()
            raise ValueError(f"Frame {frame_number} does not exist in {video_path}")

    video.release()
    return frame
