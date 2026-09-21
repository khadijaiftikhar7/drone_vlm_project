"""
Restricted zone drawing + intrusion detection.

On first run, click points on the video window to draw a polygon zone,
then press 'z' to finish it. The zone is saved to config.ZONE_FILE and
auto-loaded on future runs (press 'r' during a run to redraw/reset it).
"""

import json
import os
import cv2

import config


class RestrictedZone:
    def __init__(self):
        self.points = []          # list of (x, y) while drawing
        self.polygon = None       # finalized list of (x, y), or None if not set
        self.drawing = False
        self._load()

    def _load(self):
        if os.path.exists(config.ZONE_FILE):
            try:
                with open(config.ZONE_FILE, "r") as f:
                    data = json.load(f)
                    self.polygon = [tuple(p) for p in data.get("polygon", [])] or None
            except (json.JSONDecodeError, OSError):
                self.polygon = None

    def _save(self):
        os.makedirs(os.path.dirname(config.ZONE_FILE), exist_ok=True)
        with open(config.ZONE_FILE, "w") as f:
            json.dump({"polygon": self.polygon}, f)

    def start_drawing(self):
        self.points = []
        self.polygon = None
        self.drawing = True

    def finish_drawing(self):
        if len(self.points) >= 3:
            self.polygon = list(self.points)
            self._save()
        self.drawing = False
        self.points = []

    def mouse_callback(self, event, x, y, flags, param):
        if self.drawing and event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def contains(self, cx, cy):
        """Point-in-polygon test (ray casting). Returns False if no zone set."""
        if not self.polygon:
            return False
        n = len(self.polygon)
        inside = False
        px, py = cx, cy
        x1, y1 = self.polygon[0]
        for i in range(1, n + 1):
            x2, y2 = self.polygon[i % n]
            if py > min(y1, y2):
                if py <= max(y1, y2):
                    if px <= max(x1, x2):
                        if y1 != y2:
                            xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                        if x1 == x2 or px <= xinters:
                            inside = not inside
            x1, y1 = x2, y2
        return inside

    def draw(self, frame):
        """Draws the finalized polygon (green) or in-progress points (yellow)."""
        if self.polygon:
            pts = self.polygon
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (0, 200, 0), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self._np_pts(pts)], (0, 200, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        elif self.drawing and self.points:
            for p in self.points:
                cv2.circle(frame, p, 4, (0, 255, 255), -1)
            for i in range(len(self.points) - 1):
                cv2.line(frame, self.points[i], self.points[i + 1], (0, 255, 255), 1)

    @staticmethod
    def _np_pts(pts):
        import numpy as np
        return np.array(pts, dtype=int)
