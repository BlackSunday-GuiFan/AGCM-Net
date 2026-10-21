"""AGCM-Net: Adaptive Gating and Connected-Component Merging for
high-resolution aerial maritime small-object detection.

Modules:
  GF    - Adaptive Gated Filtering (gate net + scoring + thresholding)
  CCM   - Connected-Component Merging (group accepted tiles, rebuild regions)
  PO    - Plain Overlap tiling (fixed/vanilla overlap, SAHI-like)
  APO   - Adaptive content-aware overlap (expand only toward target side)
  DET   - Detector adapters (ultralytics YOLO / PaddleDetection PicoDet /
          mmdetection RTMDet) behind one interface
  Eval  - COCO official protocol + img/s throughput
"""
from .config import AGCMConfig, DEFAULTS
from .gate import GateFilter
from .ccm import ConnectedComponentMerger
from .overlay import PlainOverlap, AdaptiveOverlap
from .pipeline import AGCMPipeline, run_agcm_image
from .eval import evaluate_coco, EvalMetrics

__version__ = '1.0.0'
__all__ = [
    'AGCMConfig', 'DEFAULTS',
    'GateFilter', 'ConnectedComponentMerger',
    'PlainOverlap', 'AdaptiveOverlap',
    'AGCMPipeline', 'run_agcm_image',
    'evaluate_coco', 'EvalMetrics',
]