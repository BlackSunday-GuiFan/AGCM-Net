"""Base detector interface for AGCM-Net.

A Detector detects on individual crops (square or rect) and returns boxes in
the ORIGINAL image coordinate space, given the crop's top-left offset.

Boxes are dicts: {'bbox': [x, y, w, h], 'class': int, 'confidence': float}
(class is 0-based; caller maps to 1-based COCO category when building results).

Batch support (D3): detect_jobs_batch preps all pads, runs ONE model forward
over the batch (subclass _forward_batch), then maps boxes back per crop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Detector(ABC):
    """Uniform interface: detect crops -> boxes in original-image coords.

    resize_mode:
      'letterbox' - aspect-preserving scale + gray pad (YOLO/RTMDet eval)
      'stretch'   - direct non-uniform resize (PaddleDetection keep_ratio=False)
    """

    name = 'base'

    def __init__(self, device='cuda:0', resize_mode='letterbox'):
        self.device = device
        self.resize_mode = resize_mode

    @abstractmethod
    def _forward(self, pad: np.ndarray):
        """Detect on ONE HxWx3 BGR pad; returns ndarray (N,6) [x1,y1,x2,y2,s,c]."""

    def _forward_batch(self, pads):
        """Default batch = loop (subclass may override for real batching)."""
        return [self._forward(p) for p in pads]

    # ---------------------------------------------------------------
    def _to_pad(self, crop: np.ndarray):
        """Resize crop to det input per resize_mode; returns (pad, sx, sy)."""
        Hc, Wc = crop.shape[:2]
        S = self.imsz
        if self.resize_mode == 'stretch':
            ri = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
            return ri, Wc / S, Hc / S
        scale = S / max(Wc, Hc)
        nw, nh = max(1, int(Wc * scale)), max(1, int(Hc * scale))
        ri = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        pad = np.full((S, S, 3), 114, dtype=np.uint8)
        pad[:nh, :nw] = ri
        return pad, Wc / nw, Hc / nh

    def detect_crop(self, crop: np.ndarray, ox: int, oy: int):
        """Single-crop path (kept for compatibility / per-crop debugging)."""
        pad, sx, sy = self._to_pad(crop)
        det = self._forward(pad)
        return self._map(det, ox, oy, sx, sy)

    @staticmethod
    def _map(det, ox, oy, sx, sy, meta=None):
        out = []
        for k in range(len(det)):
            x1 = det[k, 0] * sx + ox
            y1 = det[k, 1] * sy + oy
            x2 = det[k, 2] * sx + ox
            y2 = det[k, 3] * sy + oy
            b = {'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                 'class': int(det[k, 5]), 'confidence': float(det[k, 4])}
            if meta is not None:
                b['_meta'] = meta
            out.append(b)
        return out

    # ---------------------------------------------------------------
    def detect_jobs_batch(self, jobs):
        """jobs: list of (crop, ox, oy[, meta]) -> flat box list, ONE batched
        forward. meta (optional dict, e.g. crop kind/tile coords) is attached
        to every box as '_meta' for source-aware postprocessing."""
        if not jobs:
            return []
        pads, metas = [], []
        for job in jobs:
            crop, ox, oy = job[0], job[1], job[2]
            meta = job[3] if len(job) > 3 else None
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            pad, sx, sy = self._to_pad(crop)
            pads.append(pad)
            metas.append((ox, oy, sx, sy, meta))
        if not pads:
            return []
        dets = self._forward_batch(pads)              # list of (Ni,6)
        boxes = []
        for det, (ox, oy, sx, sy, meta) in zip(dets, metas):
            boxes.extend(self._map(det, ox, oy, sx, sy, meta))
        return boxes

    # convenience alias
    detect_jobs = detect_jobs_batch