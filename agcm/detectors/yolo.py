"""Ultralytics YOLO detector adapter (YOLOv12n etc.)."""
from __future__ import annotations

import numpy as np
import torch

from .base import Detector


class YOLODetector(Detector):
    name = 'yolo'

    def __init__(self, weight: str, imsz: int = 640, conf: float = 0.001,
                 device: str = 'cuda:0'):
        super().__init__(device)
        from ultralytics import YOLO
        # deterministic inference: cuDNN non-determinism made the same image
        # give slightly different boxes across processes (conf 0.93 vs 0.92,
        # full boat vs half boat after NMS), breaking reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.model = YOLO(weight)
        self.imsz = imsz
        self.conf = conf

    def _forward(self, pad: np.ndarray):
        return self._forward_batch([pad])[0]

    def _forward_batch(self, pads):
        """True batch: ultralytics accepts a list of BGR images in one call."""
        if not pads:
            return []
        results = self.model(list(pads), imgsz=self.imsz, conf=self.conf,
                             iou=0.5, verbose=False, device=self.device)
        outs = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                outs.append(np.empty((0, 6)))
                continue
            bb = res.boxes.xyxy.cpu().numpy()
            cc = res.boxes.cls.cpu().numpy()[:, None]
            ss = res.boxes.conf.cpu().numpy()[:, None]
            outs.append(np.concatenate([bb, ss, cc], axis=1))
        return outs