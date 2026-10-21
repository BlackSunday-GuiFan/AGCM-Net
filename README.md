# AGCM-Net

**Adaptive Gating and Connected-Component Merging Network for Small-Object Detection in High-Resolution Aerial Maritime Imagery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)]()

AGCM-Net is a **detector-agnostic inference framework** for detecting small objects in high-resolution (up to 4K) aerial maritime imagery. It reorganizes tiled inference into a *filter → reassemble → detect* pipeline: a lightweight gating network rejects ~80% of empty tiles, a connected-component merger regroups adjacent accepted tiles into complete regions *before* detection (so cross-tile objects are never split by non-maximum suppression), and a content-aware one-sided expansion restores the borders that gating discards.

On SeaDronesSee, AGCM-Net raises YOLOv12n from 0.3071 to **0.5542 AP<sub>50–95</sub>** and large-object AP from 0.4980 to 0.6706, cutting the cross-tile fragmentation rate from 41.6% to 2.9% at **35.0 FPS** on 4K frames.

---

## Overview

| Module | Purpose |
|---|---|
| `GF` (Gated Filtering) | Per-tile binary gate (224×224) rejects ~80% empty tiles |
| `CCM` (Connected-Component Merging) | Groups adjacent accepted tiles into complete regions *before* detection |
| `APO` (Adaptive One-sided Expansion) | Expands a tile only toward sides where the neighbor tile was accepted |
| `DET` | Detector adapters (Ultralytics YOLO / PaddleDetection PicoDet / mmdetection RTMDet / DEIMv2) behind one interface |
| `Eval` | COCO official protocol (catIds, maxDets) + throughput |

### Pipeline

```
full image ──► grid tiling (adaptive, ±20% of tile size)
     │
     ▼
GF: score every tile (gate net, 224×224) ──► accept ~20% tiles
     │
     ▼
CCM: connected components of accepted tiles ──► merged regions (large targets)
     │
     ▼
APO: every accepted tile expands toward accepted neighbors (one-sided, 20%)
     │
     ▼
batch detect (accepted tiles + merged regions) ──► NMS / cross-crop fusion
```

The merge-then-detect ordering is the opposite of SAHI's detect-then-merge: NMS never has to choose between two partial boxes because the object is presented to the detector as a whole.

## Installation

```bash
git clone https://github.com/<your-name>/AGCM-Net.git
cd AGCM-Net
pip install -r requirements.txt
pip install -e .
```

Core dependencies: `torch`, `ultralytics`, `numpy`, `opencv-python`, `pycocotools`.

> PicoDet / RTMDet / DEIMv2 adapters require their own frameworks (PaddleDetection / mmdetection / DEIMv2 repo) and are optional. The core pipeline runs with the Ultralytics YOLO adapter alone.

## Data

AGCM-Net is evaluated on maritime / aerial datasets organized as COCO-format annotations plus an image directory. Point the framework at your data:

```bash
export AGCM_DATA_ROOT=/path/to/dataset   # expects images/val + annotations/instances_val.json
export AGCM_MODELS_ROOT=/path/to/weights # expects gate/ + yolo12n/ (see models/README.md)
```

### Model weights

Checkpoints (gate + tile-trained detector) are **not shipped in this repo** due to GitHub size limits. See [models/README.md](models/README.md) for the exact file layout and reproduction commands.

## Quick start

```python
import cv2
from agcm import AGCMConfig, GateFilter, AGCMPipeline
from agcm.detectors import YOLODetector

cfg = AGCMConfig()                                   # tile=640, gate=224, thr=0.5
gate = GateFilter(cfg.DEFAULTS['gate_weight'], cfg, device='cuda:0')
det = YOLODetector(cfg.DEFAULTS['yolo12n_tile'], imsz=640, device='cuda:0')

img = cv2.imread('/path/to/4k_frame.jpg')            # BGR, any resolution
pipe = AGCMPipeline(gate, det, cfg)
boxes = pipe(img)                                    # [{'bbox':[x,y,w,h], 'class':int, 'confidence':float}]
```

Or run the CLI experiment runner:

```bash
python scripts/run_experiment.py --det yolo12n_tile --mode agcm --limit 300
python scripts/run_experiment.py --det yolo12n_full --mode full --limit 300
```

## Evaluation protocol

All reported numbers follow the project-wide protocol:

- COCO official evaluation with `catIds=[1..5]`, `maxDets=[100,300,1000]`
- detector `conf=0.001`, gate threshold `thr=0.5`, tile 640
- reports COCO mAP + throughput (img/s) + per-stage latency

## Citing

If you use AGCM-Net in your research, please cite:

```bibtex
@article{agcmnet,
  title   = {Adaptive Gating and Connected-Component Merging Network for Small-Object Detection in High-Resolution Aerial Maritime Imagery},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## License

AGCM-Net is released under the [MIT License](LICENSE).
