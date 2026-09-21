"""
Utility functions for calculating drone metrics:
- Speed (pixels per frame)
- Distance from camera center
- Distance to restricted zone
- Time to reach restricted zone
"""

import math


def calculate_speed(vx, vy, fps=30):
    """
    Calculate speed from velocity components.
    
    Args:
        vx, vy: velocity in pixels per frame
        fps: frames per second (for converting to pixels per second)
    
    Returns:
        speed in pixels per second
    """
    speed_per_frame = math.sqrt(vx**2 + vy**2)
    speed_per_sec = speed_per_frame * fps
    return speed_per_sec


def calculate_distance_from_camera(cx, cy, frame_width, frame_height):
    """
    Calculate distance from drone to camera center.
    
    Args:
        cx, cy: drone center position in pixels
        frame_width, frame_height: video frame dimensions
    
    Returns:
        distance in pixels
    """
    camera_x = frame_width / 2
    camera_y = frame_height / 2
    distance = math.sqrt((cx - camera_x)**2 + (cy - camera_y)**2)
    return distance


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Calculate perpendicular distance from point (px, py) to line segment (x1,y1)-(x2,y2).
    
    Args:
        px, py: point coordinates
        x1, y1, x2, y2: line segment endpoints
    
    Returns:
        distance in pixels
    """
    # Vector from line start to point
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Line segment is a point
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Parameter t of closest point on line segment
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance
    distance = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    return distance


def distance_to_polygon(cx, cy, polygon):
    """
    Calculate minimum distance from point to polygon boundary.
    
    Args:
        cx, cy: point coordinates
        polygon: list of (x, y) tuples forming a closed polygon
    
    Returns:
        distance in pixels (0 if inside polygon)
    """
    if not polygon or len(polygon) < 2:
        return float('inf')
    
    # If point is inside polygon, return 0
    n = len(polygon)
    inside = False
    px, py = cx, cy
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if py > min(y1, y2):
            if py <= max(y1, y2):
                if px <= max(x1, x2):
                    if y1 != y2:
                        xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or px <= xinters:
                        inside = not inside
        x1, y1 = x2, y2
    
    if inside:
        return 0.0
    
    # Find minimum distance to any edge
    min_distance = float('inf')
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        dist = point_to_line_distance(cx, cy, x1, y1, x2, y2)
        min_distance = min(min_distance, dist)
    
    return min_distance


def calculate_time_to_zone(cx, cy, vx, vy, polygon, fps=30):
    """
    Calculate time (in seconds) until drone reaches restricted zone.
    
    Args:
        cx, cy: current drone center position
        vx, vy: velocity in pixels per frame
        polygon: list of (x, y) tuples forming restricted zone
        fps: frames per second
    
    Returns:
        time in seconds, or None if drone is not moving toward zone
    """
    if not polygon or len(polygon) < 2:
        return None
    
    # Current distance to zone
    current_distance = distance_to_polygon(cx, cy, polygon)
    
    # If already inside zone
    if current_distance == 0:
        return 0.0
    
    # Speed in pixels per frame
    speed_per_frame = math.sqrt(vx**2 + vy**2)
    
    # If not moving
    if speed_per_frame < 0.1:
        return None
    
    # Direction to closest point on polygon
    # For simplicity, we estimate by checking if drone is getting closer
    # Move one frame forward
    future_cx = cx + vx
    future_cy = cy + vy
    future_distance = distance_to_polygon(future_cx, future_cy, polygon)
    
    # Check if moving toward zone
    if future_distance >= current_distance:
        # Moving away or perpendicular
        return None
    
    # Estimate frames to collision
    frames_to_collision = current_distance / speed_per_frame
    
    # Convert to seconds
    time_to_collision = frames_to_collision / fps
    
    return time_to_collision


def format_speed(speed_pps):
    """Format speed in pixels per second as a readable string."""
    if speed_pps < 1:
        return f"{speed_pps:.1f} px/s"
    else:
        return f"{speed_pps:.1f} px/s"


def format_distance(distance):
    """Format distance in pixels as a readable string."""
    return f"{distance:.1f} px"


def format_time(seconds):
    """Format time in seconds as a readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    else:
        return f"{seconds:.1f} s"
