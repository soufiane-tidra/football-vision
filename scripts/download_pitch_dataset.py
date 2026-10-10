"""Download the pitch keypoint dataset from Roboflow (YOLO-pose format).

Setup (once):
    pip install roboflow
    put your key in .env:   ROBOFLOW_API_KEY=xxxxxxxx

Run:
    python -m scripts.download_pitch_dataset
"""

from src.data.roboflow_dataset import download_roboflow_dataset
from src.utils.config import load_config


def main():
    config = load_config().pitch_keypoints

    yaml_path, data = download_roboflow_dataset(
        config.workspace, config.project, config.version, config.dataset_dir
    )

    print(f"Keypoint shape: {data.get('kpt_shape')} | flip_idx: {'yes' if 'flip_idx' in data else 'no'}")
    print(f"Saved: {yaml_path}")
    print()
    print("Next: python -m scripts.check_pitch_dataset")


if __name__ == "__main__":
    main()
