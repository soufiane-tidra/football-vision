# FootballVision

[![CI](https://github.com/soufiane-tidra/football-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/soufiane-tidra/football-vision/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**AI-powered football match analysis: from broadcast video to player positions, speeds and heatmaps in real-world units.**

![FootballVision demo](docs/assets/demo.gif)

*PSG vs Bayern, tactical camera. Players are detected, tracked, split into teams by jersey color and projected onto a 2D pitch in meters, with live speed in km/h.*

> 🚧 **Status: working prototype, under active development.** See the [roadmap](#roadmap).

---

## What it does

| Step | Technique | Output |
|---|---|---|
| 1. Detection | YOLO11 (Ultralytics) | Bounding boxes for every person on screen |
| 2. Tracking | ByteTrack | Persistent ID per player across frames |
| 3. Team classification | Jersey color (Lab space, grass masked) + K-means, outlier rejection | Team A / Team B / other (referees, staff) |
| 4. Pitch calibration | Multi-landmark homography (RANSAC), interactive calibration tool, reprojection + leave-one-out validation | Pixel → meter mapping |
| 5. Camera tracking | Sparse optical flow on the pitch + ECC alignment to detected pitch lines, keyframe interpolation | One homography per frame for a panning / zooming camera |
| 6. Movement analytics | Foot-point projection, gap filling, smoothing, physical outlier filtering | Distance (m), speed (km/h), max speed |
| 7. Visualization | OpenCV | Annotated video, live 2D minimap, team heatmaps |

## Results

**Calibration**: predicted pitch lines (red) projected from 15 clicked landmarks onto the frame. Mean reprojection error **0.21 m**, leave-one-out error **0.31 m**.

![Calibration overlay](docs/assets/calibration_overlay.jpg)

**Annotated frame**: team colors, track ID, live speed, 2D tactical minimap.

![Annotated frame](docs/assets/demo_frame.jpg)

**Team heatmaps** over the demo sequence:

![Team heatmaps](docs/assets/heatmaps.png)

## How it works

```text
match.mp4
   │
   ├─► YOLO11 + ByteTrack ───────────────► tracks.csv (boxes + IDs per frame)
   │
   ├─► Pitch calibration (keyframes) ────► image ↔ pitch homography
   │        └─ camera motion + line alignment ─► homography for every frame
   │
   └─► Feet position ──homography──► (x, y) in meters
             │
             ├─► smoothing ─► distance (m), speed (km/h)
             ├─► jersey colors ─► K-means ─► teams
             └─► annotated video · minimap · heatmaps · stats
```

**Why the feet and not the box center?** The homography maps the *ground plane*. The center of a player's box is about 0.9 m above the ground, which shifts positions by several meters at this camera angle.

**Why smoothing?** About 10 cm of detection jitter per frame at 30 fps would read as 3 m/s of fake speed. Positions are smoothed before differentiating, and physically impossible steps (> 12 m/s) are discarded as tracking errors.

## Quick start

```bash
git clone https://github.com/soufiane-tidra/football-vision.git
cd football-vision
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put a match video at `data/raw/match.mp4` (paths and parameters live in [`configs/default.yaml`](configs/default.yaml)), then:

```bash
python -m scripts.track                  # YOLO + ByteTrack -> data/processed/tracks.csv
python -m scripts.calibrate_pitch        # click pitch landmarks -> pitch_calibration.json
python -m scripts.test_pitch_mapper      # validate the calibration (errors + overlay)
python -m scripts.compute_homographies   # per-frame homographies for a moving camera
python -m scripts.analyze_movement       # distance / speed per player
python -m scripts.make_demo              # annotated video, GIF, heatmaps, stats
```

### Development

```bash
pip install -r requirements-dev.txt
pytest          # unit tests: homography, landmarks, movement, calibration I/O, teams, config
ruff check .    # lint
```

Tests and lint run automatically on every push with GitHub Actions.

### Calibration tool

`scripts/calibrate_pitch.py` shows the frame next to a 2D pitch map:

- choose a standard landmark (penalty box corner, center spot, …) with `N` / `P`; its coordinates are filled in automatically
- click it in the frame using the built-in magnifier
- predicted pitch lines are drawn live, with the per-point error in meters
- several keyframes are supported (`--frame 600`)

## Project structure

```text
football-vision/
├── src/
│   ├── video/            video info, frame extraction
│   ├── detection/        YOLO detector + ByteTrack tracking
│   ├── tracking/         track data model, CSV loader
│   ├── pitch/            landmarks, homography, calibration I/O,
│   │                     camera motion, projection, drawing
│   ├── classification/   team classification (jersey colors)
│   ├── analytics/        distance, speed
│   ├── visualization/    player markers, minimap, heatmaps
│   └── utils/            configuration loading
├── scripts/              runnable entry points (python -m scripts.<name>)
├── configs/              YAML configuration (paths, pitch size, thresholds)
├── tests/                pytest unit tests
├── .github/workflows/    CI (lint + tests)
├── docs/assets/          README images
└── data/                 raw video, processed outputs (not versioned)
```

## Current limitations

- Generic COCO YOLO model: `person` also includes staff and cameramen (filtered by pitch position and color outliers).
- Track IDs fragment when players are occluded, and there is no re-identification yet.
- Pitch calibration needs manual keyframes. Camera tracking drifts on long midfield pans, so the demo uses the verified 10-second segment.
- No ball tracking yet.

## Roadmap

- [x] Video processing and frame extraction
- [x] Player detection (YOLO11) and multi-object tracking (ByteTrack)
- [x] Multi-landmark pitch calibration with validation
- [x] Per-frame homography for a moving camera
- [x] Real-world distance and speed (m, km/h)
- [x] Team classification (unsupervised, jersey colors)
- [x] Annotated video, 2D minimap, heatmaps
- [x] YAML configuration, unit tests, CI (GitHub Actions)
- [ ] Automatic pitch calibration: keypoint detection model (YOLO-pose, 32 pitch keypoints)
- [ ] Football-specific detector (player / goalkeeper / referee / ball)
- [ ] Track stitching and re-identification
- [ ] Ball tracking, possession, pass detection, passing networks
- [ ] Sprints, accelerations, team shape and formation analysis
- [ ] PostgreSQL + FastAPI backend, Streamlit dashboard
- [ ] Docker, MLflow, integration tests

## Tech stack

**Computer vision:** Python, OpenCV, NumPy, Ultralytics YOLO11, ByteTrack, homography / RANSAC, optical flow, ECC image alignment, K-means  
**Engineering:** pytest, ruff, GitHub Actions, YAML config  
**Planned:** PyTorch training, FastAPI, PostgreSQL, Streamlit, Plotly, Docker, MLflow

## License

MIT, see [LICENSE](LICENSE).
