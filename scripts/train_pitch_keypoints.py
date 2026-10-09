"""Train the pitch keypoint model (YOLO-pose, 32 keypoints) on the local GPU.

Run:
    python -m scripts.train_pitch_keypoints
    python -m scripts.train_pitch_keypoints --epochs 5      (quick test run)

Output:
    runs/pitch_keypoints/train/          curves, metrics, sample predictions
    models/pitch_keypoints.pt            best weights (used by compute_homographies)
"""

import argparse
import shutil
from pathlib import Path

import yaml

from src.utils.config import load_config


def parse_args():
    config = load_config().pitch_keypoints

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.base_model)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--imgsz", type=int, default=config.imgsz)
    parser.add_argument("--batch", type=int, default=config.batch)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    parser.add_argument("--name", default="train", help="run folder name under runs/pitch_keypoints")
    parser.add_argument("--output", default=config.weights, help="where to copy the best weights")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config().pitch_keypoints

    data_yaml = Path(config.dataset_dir) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found. Run: python -m scripts.download_pitch_dataset")

    with open(data_yaml) as file:
        data = yaml.safe_load(file)

    from ultralytics import YOLO

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path("runs/pitch_keypoints").resolve()),   # relative paths get nested by ultralytics
        name=args.name,
        exist_ok=True,
        patience=100,              # pose accuracy keeps improving long after the box converges
        # Pitch keypoints depend on the whole image layout: no mosaic tiling.
        mosaic=0.0,
        # A horizontal flip swaps left/right keypoints; only safe if the dataset says how.
        fliplr=0.5 if "flip_idx" in data else 0.0,
        plots=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, target)

    print()
    print(f"Best weights: {best}")
    print(f"Copied to:    {target}")
    print("Next: python -m scripts.compute_homographies --method keypoints")


if __name__ == "__main__":
    main()
