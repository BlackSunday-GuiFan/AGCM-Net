"""Global configuration for AGCM-Net (paths, grid, thresholds, protocols)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Repository-local data/model roots (no hard-coded user paths).
# Point these at your dataset/checkpoint locations via environment
# variables when they live outside the repo.
DATA_ROOT = Path(os.environ.get('AGCM_DATA_ROOT', str(BASE / 'data')))
MODELS_ROOT = Path(os.environ.get('AGCM_MODELS_ROOT', str(BASE / 'models')))


@dataclass
class AGCMConfig:
    """All tunable knobs of the AGCM-Net pipeline.

    tile      : side length of the basic grid tile (px)
    gate_size : gate net input side (px)
    thr       : gate accept threshold
    det_imsz  : detector input side for crop images (px)
    conf      : detector score threshold
    nms_iou   : final full-image NMS IoU
    overlap   : fixed overlap ratio for PO tiling (stride = tile*(1-overlap))
    expand    : APO one-sided expansion fraction of tile (0.2 = 128px @640)
    """
    # --- data / protocol (overridable per experiment) ---
    img_dir: Path = DATA_ROOT / 'images/val'
    coco_path: Path = DATA_ROOT / 'annotations/instances_val.json'

    # --- grid / gating ---
    tile: int = 640
    tile_adaptive: bool = True   # pick W/H-divisible tile (long/short sides
                                 # independently) within ±20% of `tile`; grid
                                 # becomes non-overlapping integer tiling
    tile_lo: float = 0.8         # adaptive tile lower bound (fraction of tile)
    tile_hi: float = 1.2         # adaptive tile upper bound (fraction of tile)
    gate_size: int = 224
    thr: float = 0.5
    overlap: float = 0.0            # PO plain-overlap ratio
    expand: float = 0.2             # APO one-sided expansion (fraction of tile)
    cut_thr: float = 0.5            # multi-cut gate: border-cut probability
                                    # threshold to expand APO on that side
    apo_mode: str = 'auto'          # APO expansion driver: 'neighbor' (D1
                                    # neighbour rule) | 'cut' (multi-cut gate
                                    # cut heads) | 'mix' (cut OR neighbour) |
                                    # 'auto' (= mix with a multi-cut gate,
                                    # else neighbor)

    # --- detection ---
    det_imsz: int = 640
    conf: float = 0.001
    nms_iou: float = 0.5
    n_cls: int = 5          # detector class count for full-image NMS (VisDrone=10)
    postprocess: str = 'nms'   # 'nms' suppress | 'nmm' merge (SAHI GREEDYNMM)
    # Cross-crop fusion for large boxes: dataset-specific gain
    # (MARIS sparse large objects +AP; SeaDronesSee dense -0.7~-1.5),
    # so OFF by default; enable via fuse_large=True for sparse scenes.
    fuse_large: bool = False
    fuse_mode: str = 'wbf'   # 'wbf' weighted-mean fusion | 'union' GREEDYNMM
                             # merge (recovers full boxes from truncated
                             # cross-tile detections that outscore the full
                             # CCM box in plain NMS)
    fuse_area: float = 20000.0   # min box area (px²) to fuse (validated 20000)
    fuse_center: float = 256.0   # max center distance (px) for fusion merge
    fuse_strategy: str = 'conf'  # wbf fusion: 'conf' classic | 'area' | 'maxbox'

    # --- gate input normalization (ImageNet stats, RGB order) ---
    gate_mean: tuple = (0.485, 0.456, 0.406)
    gate_std: tuple = (0.229, 0.224, 0.225)

    # --- COCO eval protocol ---
    cat_ids: tuple = (1, 2, 3, 4, 5)
    max_dets: tuple = (100, 300, 1000)

    # --- stage timers ---
    stages: tuple = ('load', 'gate_pre', 'gate_inf', 'cc', 'detect')


DEFAULTS = {
    # gate / detector checkpoints under MODELS_ROOT (see models/README.md).
    # Override via AGCM_MODELS_ROOT or by passing explicit weight paths.
    'gate_weight': str(MODELS_ROOT / 'gate' / 'gate-customtiny-224.pth'),
    'yolo12n_tile': str(MODELS_ROOT / 'yolo12n' / 'yolo12n-tile-640.pt'),
    'yolo12n_full': str(MODELS_ROOT / 'yolo12n' / 'yolo12n-full-1280.pt'),
    'yolo12n_full_640': str(MODELS_ROOT / 'yolo12n' / 'yolo12n-full-640.pt'),
}