"""Run the whole analysis on the configured video, step by step.

    video -> tracks -> pitch calibration -> identified players -> ball
          -> movement + possession analytics -> annotated video

Needs the trained models (see README):
    models/football_detector.pt   python -m scripts.train_detector
    models/pitch_keypoints.pt     python -m scripts.train_pitch_keypoints

Run:
    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --from build_players     (resume from a step)
    python -m scripts.run_pipeline --skip render_match
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.utils.config import load_config


STEPS = [
    ("track", [], "detect and track players, referees and ball candidates"),
    ("compute_homographies", ["--method", "keypoints", "--no-video"], "calibrate the pitch on every frame"),
    ("build_players", [], "teams, roles and stitched identities"),
    ("track_ball", [], "ball trajectory"),
    ("analyze_movement", [], "distance, speed and sprints"),
    ("analyze_possession", [], "possession, passes and turnovers"),
    ("render_match", [], "annotated video, GIF and heatmaps"),
]


def main():
    names = [name for name, _, _ in STEPS]

    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=names, default=names[0], help="first step to run")
    parser.add_argument("--skip", nargs="*", choices=names, default=[], help="steps to leave out")
    args = parser.parse_args()

    config = load_config()
    for weights in (config.detection.model, config.pitch_keypoints.weights):
        if not Path(weights).exists():
            raise SystemExit(f"Missing model: {weights} (see the training steps in the README)")

    selected = [step for step in STEPS[names.index(args.start):] if step[0] not in args.skip]
    started = time.time()

    for index, (name, extra, description) in enumerate(selected, start=1):
        print()
        print(f"=== [{index}/{len(selected)}] {name}: {description} ===")
        step_started = time.time()

        result = subprocess.run([sys.executable, "-m", f"scripts.{name}", *extra])

        if result.returncode != 0:
            raise SystemExit(f"Step '{name}' failed. Fix it, then resume with: "
                             f"python -m scripts.run_pipeline --from {name}")

        print(f"--- {name} done in {time.time() - step_started:.0f}s")

    print()
    print(f"Pipeline finished in {time.time() - started:.0f}s. Results are in data/processed and outputs/.")


if __name__ == "__main__":
    main()
