"""
Full, continuous training run on the ENTIRE dataset (not chunked).
This avoids the mAP instability caused by restarting the LR schedule
every few thousand images.

Starts fresh from yolo12n.pt backbone (not from the chunked round
weights, since those show degraded/unstable mAP).

Usage:
    python train_full.py

If interrupted (Ctrl+C, PC sleep, crash), you can resume from the last
checkpoint instead of starting over - see resume instructions at the
bottom of this file.
"""

from ultralytics.nn.modules import block as ultra_block
from eca import ECA
ultra_block.ECA = ECA

import ultralytics.nn.tasks as tasks
tasks.ECA = ECA

from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo12-p2eca.yaml")
    model.load("yolo12n.pt")

    model.train(
        data="data.yaml",         # full dataset, not a subset
        imgsz=512,
        batch=4,
        epochs=60,
        device=0,
        half=True,
        patience=20,               # more tolerant before stopping, since this is the real run
        freeze=9,
        workers=2,
        cache="disk",               # 18k images at 512px is a lot for RAM - disk cache is safer
        cos_lr=True,                 # smooth cosine LR decay - more stable than default step schedule
        save_period=5,                # checkpoint every 5 epochs, so you can resume if interrupted
        project="runs/drone_p2eca",
        name="full_run",
        mosaic=0.5,
        mixup=0.0,
        degrees=5.0,
        scale=0.5,
        fliplr=0.5,
    )

    metrics = model.val()
    print(metrics)

# ---------------------------------------------------------------------
# TO RESUME after an interruption (do NOT re-run this file from the top):
#
#   from ultralytics import YOLO
#   model = YOLO("runs/drone_p2eca/full_run/weights/last.pt")
#   model.train(resume=True)
#
# This continues exactly where it left off, LR schedule and all -
# unlike the chunked rounds, this preserves training state properly.
# ---------------------------------------------------------------------
