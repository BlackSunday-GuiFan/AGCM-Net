"""DEIMv2-N adapter for AGCM-Net (DETR/Transformer end-to-end paradigm).

Wraps a DEIMv2-style detector (DETR/Transformer end-to-end). DEIMv2 runs
transformer encoder-decoder on a resize-to-size crop, returns COCO-format
preds in crop-local coords; we offset them back into original-image coords.

NOTE: DEIMv2 infers per-crop (no batch path), so this adapter is truly
per-job -- included for paradigm diversity (end-to-end, NMS-free).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import Detector

# Set DEIMV2_PKG_ROOT to the package dir that exposes `DEIMv2Detector`.
_DEIMV2_PKG = os.environ.get('DEIMV2_PKG_ROOT', '/path/to/deimv2/package')
_DEIMV2_ROOT = os.environ.get('DEIMV2_ROOT', '/path/to/DEIMv2')


class DEIMv2Adapter(Detector):
    name = 'deimv2'

    def __init__(self, weight: str, imsz: int = 640, device: str = 'cuda:0',
                 resize_mode: str = 'letterbox', conf: float = 0.001,
                 amp: bool = False):
        super().__init__(device=device, resize_mode=resize_mode)
        self.imsz = imsz
        self.weight = weight
        self.conf = conf
        self.amp = amp
        sys.path.insert(0, _DEIMV2_PKG)
        sys.path.insert(0, _DEIMV2_ROOT)
        import torch
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='gloo', init_method='tcp://127.0.0.1:29901',
                rank=0, world_size=1)
        from deimv2_detector import DEIMv2Detector
        self._det = DEIMv2Detector(ckpt=str(weight), device=device, size=imsz)

    def _to_pad(self, img):
        """DEIM official eval protocol: Compose includes Resize size
        [640, 640] -> STRETCH to (imsz, imsz) square. The model is fully
        convolutional but scale-sensitive (positional/query setup tuned at
        640): evaluating the 1280-trained weight at 1280 collapses AP
        (0.10 vs 0.477 at the protocol's 640x640). Scales anisotropic:
        sx = W/imsz, sy = H/imsz."""
        import cv2
        h, w = img.shape[:2]
        rz = cv2.resize(img, (self.imsz, self.imsz),
                        interpolation=cv2.INTER_AREA)
        return rz, w / self.imsz, h / self.imsz

    def _forward(self, pad: np.ndarray):
        """Single-crop path (used by _forward_batch fallback / debugging)."""
        preds = self._det.detect(pad, 0)      # COCO format, pad-local coords
        if not preds:
            return np.empty((0, 6))
        det = []
        for p in preds:
            x, y, w, h = p['bbox']
            det.append([x, y, x + w, y + h, p['score'], int(p['category_id']) - 1])
        return np.array(det, dtype=np.float32).reshape(-1, 6)

    def _forward_batch(self, pads):
        """Real batch on the transformer: one model + postprocessor call over
        N same-size pads. Verified 3.4x faster than per-crop loop (batch=8:
        4.5 ms/img vs 15.2). Optional fp16 autocast (~20% faster at batch=4).
        Scores/labels post-filtered at self.conf; boxes are already in pad
        coords, so base._map translates them unchanged."""
        import torch
        n = len(pads)
        if n == 0:
            return []
        rgb = np.stack(pads)[..., ::-1].transpose(0, 3, 1, 2)   # BGR->RGB
        t = torch.from_numpy(rgb.copy()).float().div_(255.).to(self._det.device)
        H, W = pads[0].shape[:2]
        origs = torch.full((n, 2), W, dtype=torch.float32).to(self._det.device)
        origs[:, 1] = H
        with torch.no_grad():
            ctx = torch.autocast('cuda', dtype=torch.float16) if self.amp \
                else torch.no_grad()
            with ctx:
                outputs = self._det.model(t)
                results = self._det.postprocessor(outputs, origs)
        outs = []
        for r in results:
            s = r['scores'].cpu().numpy()
            l = r['labels'].cpu().numpy()
            b = r['boxes'].cpu().numpy()             # xyxy in pad coords
            keep = s >= self.conf
            s, l, b = s[keep], l[keep], b[keep]
            if len(s) == 0:
                outs.append(np.empty((0, 6)))
            else:
                outs.append(np.stack(
                    [b[:, 0], b[:, 1], b[:, 2], b[:, 3], s, l], axis=1).astype(
                    np.float32))
        return outs