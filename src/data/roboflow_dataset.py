"""Download datasets from Roboflow in YOLO format."""

import os
from pathlib import Path

import yaml


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


def download_roboflow_dataset(workspace, project, version, dataset_dir):
    """Download one dataset version and return (data.yaml path, its content)."""

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

    print(f"Downloading {workspace}/{project} v{version} ...")

    rf_project = Roboflow(api_key=api_key).workspace(workspace).project(project)

    try:
        rf_project.version(version).download("yolov8", location=str(dataset_dir), overwrite=True)
    except (ImportError, OSError) as error:
        # After extracting, roboflow imports ultralytics/torch only to tweak data.yaml;
        # we fix data.yaml ourselves, so a broken torch install must not stop the download.
        if not (Path(dataset_dir) / "data.yaml").exists():
            raise
        print(f"(ignored roboflow post-processing error: {type(error).__name__})")

    yaml_path, data = fix_data_yaml(dataset_dir)

    print()
    print(f"Dataset: {dataset_dir}")
    for folder in ("train", "valid", "test"):
        images = Path(dataset_dir) / folder / "images"
        if images.exists():
            print(f"  {folder:5s}: {len(list(images.iterdir()))} images")

    return yaml_path, data
