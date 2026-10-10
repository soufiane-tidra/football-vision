"""Download the football player detection dataset from Roboflow.

Classes: ball, goalkeeper, player, referee.

Run:
    python -m scripts.download_player_dataset
"""

from src.data.roboflow_dataset import download_roboflow_dataset
from src.utils.config import load_config


def main():
    config = load_config().player_detection

    yaml_path, data = download_roboflow_dataset(
        config.workspace, config.project, config.version, config.dataset_dir
    )

    print(f"Classes: {data.get('names')}")
    print(f"Saved: {yaml_path}")
    print()
    print("Next: python -m scripts.train_detector")


if __name__ == "__main__":
    main()
