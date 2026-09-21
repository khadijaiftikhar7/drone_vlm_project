# Drone Detection System — YOLO + FastVLM

Real-time drone/bird/airplane detection (YOLOv12n, standing in for YOLO-ULM
until its code is released) with FastVLM semantic descriptions and a
restricted-zone intrusion alert, similar to your DroneIntrusionSystem app.

## Project structure

```
drone_vlm_project/
├── config.py         # all settings — edit this first
├── detector.py        # YOLO wrapper
├── tracker.py         # simple centroid tracker (limits FastVLM calls)
├── zone.py             # restricted zone drawing + intrusion check
├── vlm_client.py       # FastVLM subprocess wrapper (has a stub fallback)
├── main.py              # entry point — run this
├── requirements.txt
└── logs/
    ├── detection_log.csv    # created on first run
    ├── intrusion_log.csv    # created on first run
    └── restricted_zone.json # created when you draw a zone
```

## 1. Quick start (no FastVLM download needed)

By default `config.ENABLE_FASTVLM = False`, so you can test the whole
pipeline — detection, tracking, zone drawing, intrusion logging — right
away with stub descriptions like `[stub] appears to be a drone`.

```bash
conda activate yolo
cd drone_vlm_project
pip install -r requirements.txt
python main.py
```

A webcam window opens. Controls:
- **z** — start drawing a restricted zone (click points on the video), press **z** again to close the polygon
- **r** — clear the current zone
- **q** — quit

Detections log to `logs/detection_log.csv`; zone entries log to `logs/intrusion_log.csv`.

## 2. Use your own trained YOLO model

In `config.py`:
```python
YOLO_WEIGHTS = "path/to/your/best.pt"
TARGET_CLASSES = ["drone", "bird", "plane"]  # match your model's actual class names exactly
```

## 3. Enable real FastVLM descriptions

```bash
git clone https://github.com/apple/ml-fastvlm.git
cd ml-fastvlm
pip install -e .
bash get_models.sh        # downloads checkpoints — large, be patient
```

Then in `config.py`:
```python
ENABLE_FASTVLM = True
FASTVLM_REPO_PATH = r"C:\path\to\ml-fastvlm"
FASTVLM_MODEL_PATH = r"C:\path\to\ml-fastvlm\checkpoints\fastvlm_0.5b_stage3"
```

Use the **0.5B** checkpoint first — it's the only size that will comfortably
share your 4GB Quadro P1000 with YOLO running at the same time.

## 4. Use a video file instead of webcam

```python
# config.py
VIDEO_SOURCE = "path/to/video.mp4"
```

## Performance notes

- FastVLM is only queried once per **new tracked object**, then re-queried
  every `VLM_EVERY_N_FRAMES` (default 30, ~once/sec) — not every frame.
  Querying every frame will tank your FPS badly on 4GB VRAM.
- The tracker is a simple centroid tracker, not ByteTrack/DeepSORT — it's
  enough to avoid redundant FastVLM calls, but it can lose IDs through
  occlusion. Swap it out later if you need robust re-identification.
- Once YOLO-ULM's code is publicly released, only `detector.py` needs to
  change — the rest of the pipeline (tracking, zone, VLM, logging) stays
  the same.

## Troubleshooting

- **`ultralytics` fails to find CUDA** — same Python 3.14/PyTorch CUDA
  incompatibility you hit before; make sure you're in the `yolo` conda
  env (Python 3.11).
- **FastVLM subprocess errors** — double check `FASTVLM_REPO_PATH` points
  to the folder containing `predict.py`, and that checkpoints finished
  downloading.
- **Zone not saving** — the `logs/` folder is created automatically, but
  make sure the process has write permission to the project directory.
