"""Train the football detector (ball, goalkeeper, player, referee) on the local GPU.

Run:
    python -m scripts.train_detector
    python -m scripts.train_detector --epochs 3      (quick test run)
    python -m scripts.train_detector --resume        (continue an interrupted run)

Output:
    runs/football_detector/train/        curves, metrics, sample predictions
    models/football_detector.pt          best weights (used by scripts.track)
"""

import argparse
import shutil
from pathlib import Path

from src.utils.config import load_config


def parse_args():
    config = load_config().player_detection

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.base_model)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--imgsz", type=int, default=config.imgsz)
    parser.add_argument("--batch", type=int, default=config.batch)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    parser.add_argument("--name", default="train", help="run folder name under runs/football_detector")
    parser.add_argument("--output", default=config.weights, help="where to copy the best weights")
    parser.add_argument("--workers", type=int, default=2,
                        help="data loader processes; each one needs RAM (mosaic at 1280 px is heavy)")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint of this run")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config().player_detection

    data_yaml = Path(config.dataset_dir) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found. Run: python -m scripts.download_player_dataset")

    from ultralytics import YOLO

    run_dir = Path("runs/football_detector").resolve() / args.name
    last = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last.exists():
            raise SystemExit(f"Nothing to resume: {last} not found.")
        model = YOLO(str(last))
        model.train(resume=True, workers=args.workers)
    else:
        model = YOLO(args.model)
        _train(model, args, data_yaml)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, target)

    print()
    print(f"Best weights: {best}")
    print(f"Copied to:    {target}")
    print("Next: set detection.model to this file in configs/default.yaml, then python -m scripts.track")


def _train(model, args, data_yaml):
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path("runs/football_detector").resolve()),   # relative paths get nested by ultralytics
        name=args.name,
        exist_ok=True,
        patience=30,
        workers=args.workers,
        plots=True,
    )


if __name__ == "__main__":
    main()
