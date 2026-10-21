# Model Weights

Checkpoints are **not shipped** in this repository (GitHub per-file limit).
Download or train them as follows and place them under `models/`:

```
models/
├── gate/
│   └── gate-customtiny-224.pth          # gating net, 224×224 input
└── yolo12n/
    ├── yolo12n-tile-640.pt              # tile-trained detector (AGCM main)
    ├── yolo12n-full-1280.pt             # full-image 1280 baseline
    └── yolo12n-full-640.pt              # full-image 640 baseline
```

The expected locations are resolved from `AGCM_MODELS_ROOT`
(defaults to `<repo>/models`). Alternatively pass explicit weight paths to
`GateFilter(weight, ...)` / `YOLODetector(weight, ...)`.

## Reproducing the weights

### Gate (CustomTiny, 224 input)

Train a binary per-tile gate that predicts whether a 640×640 tile contains any
target. Input patches are 224×224 resized tiles; label = 1 if the tile has any
GT box. See the gate training scripts in `scripts/` (BCEWithLogits + pos_weight,
input normalization ImageNet stats).

### Detector (YOLOv12n, tile 640)

Tile-level training paradigm (aligned train/inference distribution):

1. Slice the full-resolution training set into 640×640 tiles (grid, no overlap,
   only tiles containing GT objects kept as positives).
2. Fine-tune YOLOv12n (COCO pretrained) on the tiles at `imgsz=640`.

## Environments

- Core pipeline: any env with `torch + ultralytics` (see `requirements.txt`).
- PicoDet / RTMDet / DEIMv2 adapters additionally need their frameworks;
  they are optional and selected only when instantiated.
