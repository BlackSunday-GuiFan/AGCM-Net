"""PaddleDetection PicoDet-S adapter (runs in the paddle_det conda env).

Full-image inference needs scale_factor = [imsz/H, imsz/W] (<1) so the
post_process maps boxes back to original coords; crop inference passes
scale_factor=1 because boxes are already consumed in pad coords by base.detect_crop.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import Detector

# PaddleDetection repo root (set PADDLE_DET_ROOT to your checkout).
PADDLE_DET = Path(os.environ.get('PADDLE_DET_ROOT',
                                 '/path/to/PaddleDetection'))
CFG = PADDLE_DET / 'configs/picodet/picodet_s_640_coco_sds_eval.yml'  # eval cfg
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PicoDetDetector(Detector):
    name = 'picodet'

    def __init__(self, weight: str, imsz: int = 416, conf: float = 0.001,
                 device: str = 'cuda:0'):
        # PaddleDetection EvalReader resizes with keep_ratio=False (direct
        # stretch); PicoHead anchors match that aspect-distorted space.
        super().__init__(device, resize_mode='stretch')
        sys.path.insert(0, str(PADDLE_DET))
        import paddle
        from ppdet.core.workspace import load_config, create
        from ppdet.utils.checkpoint import load_pretrain_weight
        self.paddle = paddle
        cfg = load_config(str(CFG))
        # PicoHead pre-generates anchor points from eval_size; must match the
        # actual input so boxes/preds have consistent coordinates.
        cfg['eval_height'], cfg['eval_width'] = imsz, imsz
        cfg['eval_size'] = [imsz, imsz]
        try:
            cfg['TestReader']['inputs_def']['image_shape'] = [1, 3, imsz, imsz]
        except Exception:
            pass
        self.model = create(cfg.architecture)
        load_pretrain_weight(self.model, str(weight))
        self.model.eval()
        self.imsz = imsz
        self.conf = conf
        self._mean_t = self._std_t = None

    # ------------------------------------------------------------------
    def _pre(self, pad: np.ndarray):
        im = pad[..., ::-1].astype(np.float32) / 255.0    # BGR -> RGB
        im = (im - MEAN) / STD
        return self.paddle.to_tensor(im.transpose(2, 0, 1)[None].copy())

    def _pre_batch(self, pads):
        # uint8 -> GPU once, normalize on GPU. The old path did the whole
        # BGR->RGB / float32 / mean-std in CPU numpy (~13ms/img, batch-const);
        # flip+cast+arithmetic on the device costs ~1ms/img total.
        im = np.stack(pads).transpose(0, 3, 1, 2)         # (B,3,H,W) uint8 BGR
        x = self.paddle.to_tensor(im).cuda()              # uint8 on GPU
        x = x.astype('float32').flip(1) / 255.0           # BGR -> RGB, /255
        if self._mean_t is None:
            dev = self.paddle.CUDAPlace(0)
            self._mean_t = self.paddle.to_tensor(
                MEAN[None, :, None, None], place=dev)
            self._std_t = self.paddle.to_tensor(
                STD[None, :, None, None], place=dev)
        return (x - self._mean_t) / self._std_t

    def _forward(self, pad: np.ndarray):
        return self._forward_batch([pad])[0]

    def _forward_batch(self, pads):
        with self.paddle.no_grad():
            outs = self.model({'image': self._pre_batch(pads),
                               'scale_factor': self.paddle.to_tensor(
                                   np.ones((len(pads), 2), dtype=np.float32))})
        bbox = outs['bbox'].numpy()      # (N,6) cls,score,x1,y1,x2,y2 (PD nms)
        bnum = outs['bbox_num'].numpy()  # (B,)
        outs_list, i = [], 0
        for n in bnum:
            n = int(n)
            if n == 0:
                outs_list.append(np.empty((0, 6)))
                continue
            b = bbox[i:i + n].copy()
            i += n
            outs_list.append(np.stack(
                [b[:, 2], b[:, 3], b[:, 4], b[:, 5], b[:, 1], b[:, 0]],
                axis=1))                  # -> x1,y1,x2,y2,s,c (0-based cls)
        return outs_list