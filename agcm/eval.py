"""COCO official evaluation + throughput measurement.

Mirrors the project-wide protocol: catIds=[1..5], maxDets=[100,300,1000],
conf=0.001. EvalMetrics computes mAP50-95/50/75 and area-split APs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class EvalMetrics:
    mAP50_95: float = 0.0
    mAP50: float = 0.0
    mAP75: float = 0.0
    small: float = 0.0
    medium: float = 0.0
    large: float = 0.0
    n_pred: int = 0

    def asdict(self):
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def evaluate_coco(preds: List[dict], coco_path: str,
                  cat_ids=(1, 2, 3, 4, 5), max_dets=(100, 300, 1000),
                  subset_ids=None) -> EvalMetrics:
    """COCO official mAP on COCO-format preds (image_id/category_id/bbox/score)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(str(coco_path))
    if not preds:
        return EvalMetrics(n_pred=0)
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.params.maxDets = list(max_dets)
    ev.params.catIds = list(cat_ids)
    if subset_ids is not None:
        ev.params.imgIds = list(subset_ids)
    ev.evaluate()
    ev.accumulate()
    p = ev.eval['precision']

    def meanp(sel):
        v = p[sel]
        return float(np.mean(v[v > -1])) if np.any(v > -1) else 0.0

    return EvalMetrics(
        mAP50_95=meanp(np.s_[:, :, :, 0, 0]),
        mAP50=meanp(np.s_[0, :, :, 0, 0]),
        mAP75=meanp(np.s_[5, :, :, 0, 0]),
        small=meanp(np.s_[:, :, :, 1, 0]),
        medium=meanp(np.s_[:, :, :, 2, 0]),
        large=meanp(np.s_[:, :, :, 3, 0]),
        n_pred=len(preds),
    )


def run_full_eval(predict_fn, image_ids, coco_path, n_images=None,
                  print_stages=False):
    """Run predict_fn(img_id)->preds over the val set, measure img/s + mAP.

    predict_fn: callable(img_id) returning COCO-format preds for one image.
    Timing includes image load + pipeline; resolves to ms/img and img/s.
    """
    ids = image_ids[:n_images] if n_images else image_ids
    preds, t0 = [], time.time()
    for i, img_id in enumerate(ids):
        preds += predict_fn(img_id)
        if (i + 1) % 300 == 0:
            print(f'  [{i+1}/{len(ids)}] {time.time()-t0:.0f}s', flush=True)
    dt = time.time() - t0
    met = evaluate_coco(preds, coco_path)
    out = {'n_imgs': len(ids), 'total_s': round(dt, 2),
           'img/s': round(len(ids) / dt, 2),
           'ms/img': round(1000 * dt / len(ids), 2),
           'mAP': met.asdict()}
    return out


def save_result(result: dict, path: Path):
    path.write_text(json.dumps(result, indent=2))
    return path


def _iou(ab, gt):
    ax1, ay1, ax2, ay2 = ab
    bx1, by1, bx2, by2 = gt
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def fragmentation_metrics(preds, coco_path, area_min=96 * 96, tau=0.75,
                          iou_match=0.5):
    """Detection-level fragmentation of LARGE objects.

    o in O_L^det = large GT boxes matched by any prediction (IoU > iou_match)
    b*(o)       = prediction with the highest IoU to o
    DFR@tau     = |{o in O_L^det : IoU(b*(o), o) < tau}| / |O_L^det|
    AIoU_L      = mean over O_L^det of IoU(b*(o), o)

    Unlike the geometric GF-FR (grid-line crossings), these measure whether
    geometric fragmentation actually degraded the detection output: a large
    object whose best matched box is incomplete (IoU below tau) is counted.
    """
    from pycocotools.coco import COCO
    coco = COCO(str(coco_path))
    by_img = {}
    for p in preds:
        by_img.setdefault(p['image_id'], []).append(p)
    matched = []          # best IoU per matched large GT
    below = 0
    for im in coco.imgs.values():
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=im['id'])):
            x, y, w, h = ann['bbox']
            if w * h < area_min:
                continue
            gt = (x, y, x + w, y + h)
            best_iou = 0.0
            for p in by_img.get(im['id'], []):
                px, py, pw, ph = p['bbox']
                ov = _iou((px, py, px + pw, py + ph), gt)
                if ov > best_iou:
                    best_iou = ov
            if best_iou > iou_match:
                matched.append(best_iou)
                if best_iou < tau:
                    below += 1
    n = len(matched)
    return {
        'n_large_matched': n,
        'DFR@%.2f' % tau: (below / n if n else 0.0),
        'AIoU_L': (float(np.mean(matched)) if n else 0.0),
    }