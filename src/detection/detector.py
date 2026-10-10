import csv
from pathlib import Path

from ultralytics import YOLO


class FootballDetector:

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0]

    def track_video(
        self,
        video_path,
        csv_path="data/processed/tracks.csv",
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.25
    ):
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # stream=True yields one frame at a time instead of keeping every
        # result (and its full image) in memory.
        results = self.model.track(
            source=video_path,
            tracker=tracker,
            persist=True,
            imgsz=imgsz,
            conf=conf,
            stream=True,
            verbose=False
        )

        with open(csv_path, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "frame",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y"
            ])

            for frame_number, result in enumerate(results):

                boxes = result.boxes

                if boxes is None:
                    continue

                ids = boxes.id

                if ids is None:
                    continue

                for box, track_id, class_id, confidence in zip(
                    boxes.xyxy,
                    ids,
                    boxes.cls,
                    boxes.conf
                ):

                    x1, y1, x2, y2 = box.tolist()

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    class_id = int(class_id)
                    track_id = int(track_id)
                    confidence = float(confidence)

                    class_name = self.model.names[class_id]

                    writer.writerow([
                        frame_number,
                        track_id,
                        class_id,
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        center_x,
                        center_y
                    ])

        return csv_path