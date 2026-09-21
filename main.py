"""
Drone Detection System: YOLO + FastVLM + Restricted Zone Intrusion Detection
==============================================================================

Controls while running:
    z   - start/finish drawing a restricted zone (click points, press z again to close)
    r   - reset/clear the current zone
    q   - quit

Setup:
    1. conda activate yolo
    2. pip install -r requirements.txt
    3. (optional, for real FastVLM descriptions) follow README.md to set up
       apple/ml-fastvlm, then set ENABLE_FASTVLM = True in config.py
    4. python main.py
"""

import os
import csv
import time
import tempfile
from pathlib import Path
from datetime import datetime

import cv2

import config
from detector import Detector
from tracker import SimpleTracker
from zone import RestrictedZone
from vlm_client import query_fastvlm
from metrics import (
    calculate_speed,
    calculate_distance_from_camera,
    calculate_time_to_zone,
    format_speed,
    format_distance,
    format_time,
)


def open_csv_writer(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = Path(path).exists()
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if not exists:
        writer.writerow(header)
    return f, writer


def main():
    detector = Detector()
    tracker = SimpleTracker(config.TRACKER_MAX_DISTANCE, config.TRACKER_MAX_MISSED_FRAMES)
    zone = RestrictedZone()

    vlm_descriptions = {}     # track_id -> last description
    last_vlm_frame = {}       # track_id -> frame_idx last queried
    intruding_tracks = set()  # track_ids currently flagged as intruding

    detection_log_file, detection_writer = open_csv_writer(
        config.DETECTION_LOG_CSV,
        ["timestamp", "frame", "track_id", "yolo_class", "confidence", "fastvlm_description"],
    )
    intrusion_log_file, intrusion_writer = open_csv_writer(
        config.INTRUSION_LOG_CSV,
        ["timestamp", "frame", "track_id", "yolo_class"],
    )

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    window_name = "Drone Detection: YOLO + FastVLM"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, zone.mouse_callback)

    frame_idx = 0
    fps_timer = time.time()
    fps = 0

    print("Controls: 'z' = draw/finish zone | 'r' = reset zone | 'q' = quit")
    if not config.ENABLE_FASTVLM:
        print("NOTE: FastVLM is disabled (config.ENABLE_FASTVLM = False). "
              "Using stub descriptions. See README.md to enable real FastVLM.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detections = detector.detect(frame)
            boxes = [d[0] for d in detections]
            assigned = tracker.update(boxes, frame_idx)
            box_to_id = {v: k for k, v in assigned.items()}

            for box, cls_name, conf in detections:
                track_id = box_to_id.get(box)
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # --- Get velocity and position from tracker ---
                vx, vy = tracker.get_velocity(track_id)
                
                # --- Calculate metrics ---
                speed_pps = calculate_speed(vx, vy, fps=30)  # pixels per second
                distance_from_camera = calculate_distance_from_camera(
                    cx, cy, config.FRAME_WIDTH, config.FRAME_HEIGHT
                )
                time_to_zone = None
                if zone.polygon:
                    time_to_zone = calculate_time_to_zone(cx, cy, vx, vy, zone.polygon, fps=30)

                # --- FastVLM query (new track or periodic re-query) ---
                should_query = (
                    track_id not in vlm_descriptions
                    or frame_idx - last_vlm_frame.get(track_id, -999) >= config.VLM_EVERY_N_FRAMES
                )
                if should_query:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        crop_path = os.path.join(tmp_dir, f"crop_{track_id}_{frame_idx}.jpg")
                        cv2.imwrite(crop_path, crop)
                        description = query_fastvlm(crop_path, cls_name)
                        vlm_descriptions[track_id] = description
                        last_vlm_frame[track_id] = frame_idx

                        detection_writer.writerow([
                            datetime.now().isoformat(), frame_idx, track_id,
                            cls_name, f"{conf:.2f}", description,
                        ])
                        detection_log_file.flush()

                description = vlm_descriptions.get(track_id, "...")

                # --- Restricted zone intrusion check ---
                is_intruding = zone.contains(cx, cy)
                if is_intruding and track_id not in intruding_tracks:
                    intruding_tracks.add(track_id)
                    intrusion_writer.writerow([
                        datetime.now().isoformat(), frame_idx, track_id, cls_name,
                    ])
                    intrusion_log_file.flush()
                    print(f"[ALERT] Track {track_id} ({cls_name}) entered restricted zone at frame {frame_idx}")
                elif not is_intruding and track_id in intruding_tracks:
                    intruding_tracks.discard(track_id)

                box_color = (0, 0, 255) if is_intruding else (0, 255, 0)
                label = f"ID{track_id} {cls_name} {conf:.2f}" + (" [INTRUSION]" if is_intruding else "")

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                cv2.putText(frame, description[:60], (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                
                # --- Display speed, distance, and time to zone ---
                metric_y = y2 + 45
                speed_str = f"Speed: {format_speed(speed_pps)}"
                distance_str = f"Dist: {format_distance(distance_from_camera)}"
                time_str = f"To Zone: {format_time(time_to_zone)}"
                
                cv2.putText(frame, speed_str, (x1, metric_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 255), 1)
                cv2.putText(frame, distance_str, (x1, metric_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
                cv2.putText(frame, time_str, (x1, metric_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1)

            zone.draw(frame)

            if frame_idx % 10 == 0:
                fps = 10 / (time.time() - fps_timer)
                fps_timer = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if zone.drawing:
                cv2.putText(frame, "Drawing zone... click points, press 'z' to finish",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                if zone.drawing:
                    zone.finish_drawing()
                else:
                    zone.start_drawing()
            elif key == ord("r"):
                zone.polygon = None
                zone.drawing = False
                zone.points = []

    cap.release()
    cv2.destroyAllWindows()
    detection_log_file.close()
    intrusion_log_file.close()


if __name__ == "__main__":
    main()
