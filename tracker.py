"""
Minimal centroid-based object tracker.

Not as robust as ByteTrack/DeepSORT (no re-identification through full
occlusion), but it's dependency-free and good enough to stop FastVLM from
being re-queried every single frame for the same object. Swap in
ByteTrack later if you need ID persistence through occlusion.
"""


class SimpleTracker:
    def __init__(self, max_distance=60, max_missed_frames=60):
        self.next_id = 0
        self.tracks = {}  # id -> (cx, cy, last_seen_frame, vx, vy)
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

    def update(self, boxes, frame_idx):
        """
        boxes: list of (x1, y1, x2, y2)
        returns: dict of track_id -> box, matching the input boxes to
                 existing or newly created track IDs.
        """
        assigned = {}
        used_ids = set()

        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            best_id, best_dist = None, self.max_distance
            for tid, track_data in self.tracks.items():
                if tid in used_ids:
                    continue
                tcx, tcy = track_data[0], track_data[1]
                dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                # New track: initialize with zero velocity
                self.tracks[best_id] = (cx, cy, frame_idx, 0.0, 0.0)
            else:
                # Update existing track: calculate velocity
                old_cx, old_cy, last_frame, _, _ = self.tracks[best_id]
                frame_diff = frame_idx - last_frame if frame_idx > last_frame else 1
                vx = (cx - old_cx) / frame_diff
                vy = (cy - old_cy) / frame_diff
                self.tracks[best_id] = (cx, cy, frame_idx, vx, vy)

            used_ids.add(best_id)
            assigned[best_id] = box

        stale = [
            tid for tid, track_data in self.tracks.items()
            if frame_idx - track_data[2] > self.max_missed_frames
        ]
        for tid in stale:
            del self.tracks[tid]

        return assigned

    def get_velocity(self, track_id):
        """Returns (vx, vy) velocity for a track, or (0, 0) if not found."""
        if track_id in self.tracks:
            return (self.tracks[track_id][3], self.tracks[track_id][4])
        return (0.0, 0.0)

    def get_position(self, track_id):
        """Returns (cx, cy) position for a track, or None if not found."""
        if track_id in self.tracks:
            return (self.tracks[track_id][0], self.tracks[track_id][1])
        return None
