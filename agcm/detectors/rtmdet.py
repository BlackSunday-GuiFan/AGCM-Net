"""mmdetection RTMDet adapter (openmmlab conda env)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from .base import Detector

# mmdetection repo root (set MMDET_ROOT to your checkout).
MMDET = Path(os.environ.get('MMDET_ROOT', '/path/to/mmdetection'))
CFG = MMDET / 'configs/rtmdet/rtmdet_tiny_8xb16-300e_sds640.py'


class RTMDetDetector(Detector):
    name = 'rtmdet'

    def __init__(self, weight: str, imsz: int = 640, conf: float = 0.001,
                 device: str = 'cuda:0'):
        super().__init__(device)
        sys.path.insert(0, str(MMDET))
        import torch
        from mmengine.config import Config
        from mmdet.utils import register_all_modules
        register_all_modules()
        cfg = Config.fromfile(str(CFG))
        cfg.work_dir = '/tmp/rtmdet_agcm'
        from mmdet.registry import MODELS
        self.model = MODELS.build(cfg.model)
        ckpt = torch.load(weight, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        self.model.load_state_dict(ckpt)
        self.model.eval().cuda(0)
        self.imsz = imsz
        self.conf = conf
        from mmdet.structures import DetDataSample
        self.DetDataSample = DetDataSample
        # Pre-generate per-level grid points (offset=0, MlvlPointGenerator).
        # H sequence depends on input size: imsz/8, /16, /32.
        dev = torch.cuda.current_device()
        self._priors = {}       # imsz -> list of (num_pts, 2) float32 on GPU
        self._build_priors(imsz, dev)

    def _build_priors(self, imsz, dev):
        """Build grid-point cache for a given input size."""
        self._priors[imsz] = []
        s = 8
        for H in (imsz // 8, imsz // 16, imsz // 32):
            ys, xs = torch.meshgrid(
                torch.arange(H, device=dev),
                torch.arange(H, device=dev), indexing='ij')
            self._priors[imsz].append(
                torch.stack([xs.flatten() * s, ys.flatten() * s], -1).float())
            s *= 2
        self._priors_h = imsz   # currently cached input size

    def _forward_batch(self, pads):
        """True batch: stack pads, one model forward.

        mmdet 3.x: model expects data preprocessed by data_preprocessor.
        We invoke it manually (val_step would also call it, but that's overhead).
        Input layout: HWC uint8 BGR (matches PaddleDet/Picodet convention),
        transposed to CHW float inside data_preprocessor.
        """
        import torch
        x = torch.from_numpy(np.stack(pads)).cuda().permute(0, 3, 1, 2).contiguous()
        # x is now (N, C=3, H, W) uint8 BGR
        n = x.shape[0]
        samples = [self.DetDataSample(metainfo={
            'img_shape': (self.imsz, self.imsz),
            'ori_shape': (self.imsz, self.imsz),
            'scale_factor': (1.0, 1.0)}) for _ in range(n)]
        data = {'inputs': x, 'data_samples': samples}
        data = self.model.data_preprocessor(data, False)
        with torch.no_grad():
            out = self.model._run_forward(data, mode='tensor')
        return self._postprocess_batch(out, n)

    def _postprocess_batch(self, out, n):
        """Vectorized decode + NMS, semantically identical to mmdet
        RTMDetHead.predict_by_feat.

        Key semantics replicated:
        - filter_scores_and_topk: scores(mask>thr) -> sort desc -> topk(nms_pre)
          over the FULL (anchor, class) expanded set (per image).
        - topk happens BEFORE distance decode; only kept entries are decoded.
        - batched nms (per-image, by offset labels), max_per_img=300.
        nms_pre=30000 < 8400*5=42000 expanded entries, so topk CAN truncate and
        is therefore kept to stay bit-identical to mmdet.

        The per-image topk loop is only n iterations over 1-D sortable tensors;
        decode is one (8400,4) tensor op for the whole batch.
        """
        import torch
        cls_scores, bbox_preds = out          # tuples: (B,5,H,W) & (B,4,H,W)
        strides = (8, 16, 32)
        all_cls, all_box, all_pts = [], [], []
        for lvl, (cs, bp) in enumerate(zip(cls_scores, bbox_preds)):
            B, _, H, W = cs.shape
            st = strides[lvl]
            # (B, H*W, 5) sigmoid prob; (B, H*W, 4) distances
            s = cs.permute(0, 2, 3, 1).reshape(B, -1, cs.shape[1]).sigmoid()
            d = bp.permute(0, 2, 3, 1).reshape(B, -1, 4)
            # grid points: (x, y) centers (0.5-offset) * stride, per level
            ys, xs = torch.meshgrid(
                torch.arange(H, device=cs.device),
                torch.arange(W, device=cs.device), indexing='ij')
            pts = torch.stack(
                [(xs.flatten() + 0.5) * st, (ys.flatten() + 0.5) * st], -1)
            all_cls.append(s); all_box.append(d)
            all_pts.append(pts.expand(B, -1, 2))
        cls = torch.cat(all_cls, 1)           # (B, N=8400, 5)
        box = torch.cat(all_box, 1)           # (B, 8400, 4)
        pts = torch.cat(all_pts, 1)           # (B, 8400, 2)
        from mmdet.structures.bbox.transforms import distance2bbox
        # decode ALL anchors once (batch op) -- same math as mmdet decode
        boxes = distance2bbox(
            pts, box, max_shape=(self.imsz, self.imsz))   # (B, 8400, 4)
        thr = self.conf                       # score_thr=0.001
        nms_pre = 30000
        outs = []
        for i in range(n):
            dets_per_lvl = []
            for lvl, (cs, bp) in enumerate(
                    zip(cls_scores, bbox_preds)):   # per-level topk, like mmdet
                ncls = cs.shape[1]
                H = cs.shape[-2]
                s = cs[i].permute(1, 2, 0).reshape(-1, ncls).sigmoid()
                d = bp[i].permute(1, 2, 0).reshape(-1, 4)
                flat = s.flatten()
                keep = (flat > thr).nonzero().flatten()
                if keep.numel() == 0:
                    continue
                srt = flat[keep].sort(descending=True)
                keep = keep[srt.indices][:nms_pre]
                a_idx = keep // ncls           # anchor row
                c_idx = keep % ncls            # class
                pts = self._priors[self._priors_h][lvl][a_idx]
                bb = distance2bbox(
                    pts, d[a_idx], max_shape=(self.imsz, self.imsz))
                ss = flat[keep].unsqueeze(1)
                cc = c_idx.unsqueeze(1).float()
                dets_per_lvl.append(torch.cat([bb, ss, cc], 1))
            if not dets_per_lvl:
                outs.append(np.empty((0, 6)))
                continue
            det = torch.cat(dets_per_lvl)      # (K, 6)
            cls_ids = det[:, 5].long()
            from mmcv.ops import batched_nms
            _, keep2 = batched_nms(det[:, :4], det[:, 4], cls_ids,
                                   dict(type='nms', iou_threshold=0.65))
            det = det[keep2][:300]             # max_per_img
            outs.append(det.cpu().numpy())
        return outs

    def _forward(self, pad):
        return self._forward_batch([pad])[0]