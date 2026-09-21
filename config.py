"""
Central configuration for the Drone Detection + FastVLM project.
Edit the values below to match your machine before running main.py.
"""

# ------------------------- YOLO DETECTOR -------------------------

YOLO_WEIGHTS = r"C:\Users\khadija\.cache\huggingface\hub\models--doguilmak--Drone-Detection-YOLOv8x\snapshots\a7cae6c26939ae7bfef470acf79bb1e36301bd3b\weight\best.pt"
CONF_THRESHOLD = 0.1

# Must match the class names your YOLO model was trained with
TARGET_CLASSES = ["drone"]

# ------------------------------ VIDEO -----------------------------
VIDEO_SOURCE = r"D:\GAMES\gameplay\Screen Recording 2026-08-07 111415.mp4"        # 0 = default webcam, or a path/URL to a video file
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720# ------------------------------ FASTVLM ----------------------------
# Set ENABLE_FASTVLM = False to run the project immediately without
# downloading FastVLM checkpoints (a stub description is used instead).
ENABLE_FASTVLM = False

FASTVLM_REPO_PATH = r"C:\path\to\ml-fastvlm"
FASTVLM_MODEL_PATH = r"C:\path\to\ml-fastvlm\checkpoints\fastvlm_0.5b_stage3"
FASTVLM_PROMPT = (
    "Describe this object in one short sentence. Is it a drone, bird, "
    "or airplane? Note any distinguishing features."
)
VLM_EVERY_N_FRAMES = 30      # re-query FastVLM on a track roughly once a second at 30fps
FASTVLM_TIMEOUT_SEC = 15

# --------------------------- RESTRICTED ZONE ------------------------
# Draw the zone interactively on first run (sePe README). Points are
# saved to this file and reloaded automatically on future runs.
ZONE_FILE = "logs/restricted_zone.json"

# ------------------------------- LOGGING ----------------------------
DETECTION_LOG_CSV = "logs/detection_log.csv"
INTRUSION_LOG_CSV = "logs/intrusion_log.csv"

# Simple centroid tracker settings
TRACKER_MAX_DISTANCE = 60
TRACKER_MAX_MISSED_FRAMES = 60
