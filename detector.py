"""
YOLO detector wrapper.

Uses YOLOv12n as a stand-in for YOLO-ULM (no public code released yet as
of Aug 2026). Swap config.YOLO_WEIGHTS for your own trained model, or
replace the Detector class internals once YOLO-ULM's repo is available —
the rest of the project doesn't need to change.
"""

from ultralytics import YOLO
import torch

import config


class Detector:
    def __init__(self):
        # choose device: prefer CUDA if available and enabled in config
        device = "cuda" if (getattr(config, "USE_CUDA", False) and torch.cuda.is_available()) else "cpu"
        try:
            # ultralytics YOLO supports a device argument in recent versions
            self.model = YOLO(config.YOLO_WEIGHTS, device=device)
        except TypeError:
            # fallback: construct then move model to device
            self.model = YOLO(config.YOLO_WEIGHTS)
            if device == "cuda":
                try:
                    self.model.to(device)
                except Exception:
                    pass
        print(f"Detector using device: {device}")

    def detect(self, frame):
        """Returns list of (box, cls_name, confidence) for target classes only.
        box is (x1, y1, x2, y2) in pixel coordinates."""
        results = self.model(frame, conf=config.CONF_THRESHOLD, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls[0])]
            if cls_name not in config.TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(((x1, y1, x2, y2), cls_name, conf))

        return detections
