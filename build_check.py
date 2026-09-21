"""
Registers the custom ECA module with Ultralytics and builds the modified
yolo12-p2eca.yaml to confirm it loads with no shape-mismatch errors.

Run this BEFORE training. If model.info() prints cleanly, the architecture
is valid and you're ready to call model.train(...).
"""

# 1. Register ECA BEFORE importing/building any YOLO model
from ultralytics.nn.modules import block as ultra_block
from eca import ECA
ultra_block.ECA = ECA

import ultralytics.nn.tasks as tasks
tasks.ECA = ECA

# 2. Now build the model from the modified YAML
from ultralytics import YOLO

model = YOLO("yolo12-p2eca.yaml")   # random-init weights, architecture only
model.info(detailed=False)

# Quick forward-pass sanity check with a dummy image
import torch
dummy = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    out = model.model(dummy)
print("Forward pass OK. Number of Detect outputs:", len(out[0]) if isinstance(out, tuple) else "n/a")
