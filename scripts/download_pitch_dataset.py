"""Download the pitch keypoint dataset from Roboflow (YOLO-pose format).

Setup (once):
    pip install roboflow
    put your key in .env:   ROBOFLOW_API_KEY=xxxxxxxx

Run:
    python -m scripts.download_pitch_dataset
"""

import os
from pathlib import Path

import yaml

from src.utils.config import load_config


def read_env(path=".env"):
    values = {}
    path = Path(path)

    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def fix_data_yaml(dataset_dir):
    """Roboflow writes '../train/images'-style paths; make them relative to the dataset folder."""

    dataset_dir = Path(dataset_dir).resolve()
    yaml_path = dataset_dir / "data.yaml"

    with open(yaml_path) as file:
        data = yaml.safe_load(file)

    data["path"] = str(dataset_dir)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (dataset_dir / folder / "images").exists():
            data[split] = f"{folder}/images"
        else:
            data.pop(split, None)

    with open(yaml_path, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    return yaml_path, data


def main():
    config = load_config().pitch_keypoints

    api_key = os.environ.get("ROBOFLOW_API_KEY") or read_env().get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY not found. Add this line to the .env file in the project folder:\n"
            "ROBOFLOW_API_KEY=your_key"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("The roboflow package is missing. Run: pip install roboflow")

    print(f"Downloading {config.workspace}/{config.project} v{config.version} ...")

    project = Roboflow(api_key=api_key).workspace(config.workspace).project(config.project)

    try:
        project.version(config.version).download("yolov8", location=config.dataset_dir, overwrite=True)
    except (ImportError, OSError) as error:
        # After extracting, roboflow imports ultralytics/torch only to tweak data.yaml;
        # we fix data.yaml ourselves, so a broken torch install must not stop the download.
        if not (Path(config.dataset_dir) / "data.yaml").exists():
            raise
        print(f"(ignored roboflow post-processing error: {type(error).__name__})")

    yaml_path, data = fix_data_yaml(config.dataset_dir)

    print()
    print(f"Dataset: {config.dataset_dir}")
    for folder in ("train", "valid", "test"):
        images = Path(config.dataset_dir) / folder / "images"
        if images.exists():
            print(f"  {folder:5s}: {len(list(images.iterdir()))} images")

    print(f"Keypoint shape: {data.get('kpt_shape')} | flip_idx: {'yes' if 'flip_idx' in data else 'no'}")
    print(f"Saved: {yaml_path}")
    print()
    print("Next: python -m scripts.check_pitch_dataset")


if __name__ == "__main__":
    main()
