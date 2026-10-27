"""AGCM-Net experiment runner.

Evaluates the AGCM pipeline (or Full-Image / SAHI baselines) on a COCO-format
dataset and reports official COCO mAP + throughput + per-stage latency.

Usage:
  python scripts/run_experiment.py --det yolo12n_tile --mode agcm --limit 300
  python scripts/run_experiment.py --det yolo12n_full --mode full --limit 300
  python scripts/run_experiment.py --det yolo12n_tile --mode sahi

Data is located via the AGCM_DATA_ROOT environment variable
(expects <root>/images/val + <root>/annotations/instances_val.json);
weights via AGCM_MODELS_ROOT (see models/README.md). Override with --img-dir,
--coco and --gate / explicit --weights.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'agcm'))

import cv2
import numpy as np

from agcm import (AGCMConfig, GateFilter, AGCMPipeline, evaluate_coco)
from agcm.config import DATA_ROOT, MODELS_ROOT
from agcm.detectors import YOLODetector
from agcm.overlay import plain_grid

# ----------------------------------------------------------------------
DETECTORS = {
    'yolo12n_full': dict(cls=YOLODetector, w='yolo12n_full', imsz=1280),
    'yolo12n_full_640': dict(cls=YOLODetector, w='yolo12n_full_640', imsz=640),
    'yolo12n_tile': dict(cls=YOLODetector, w='yolo12n_tile', imsz=640),
}

# dataset protocol registry: {name: (cat_ids, n_cls)}
PROTOCOLS = {
    'default': ((0, 1, 2, 3, 4), 5),
    'visdrone': (tuple(range(1, 11)), 10),
    'maris': ((1, 2, 3), 3),
}


def build_det(name, imsz, conf=None, device=None, weights=None):
    ent = DETECTORS[name]
    from agcm.config import DEFAULTS
    w = weights or ent.get('weights') or DEFAULTS[ent['w']]
    kw = dict(imsz=imsz if imsz else ent['imsz'])
    if conf is not None:
        kw['conf'] = conf
    if device is not None:
        kw['device'] = device
    if ent.get('resize_mode'):
        kw['resize_mode'] = ent['resize_mode']
    return ent['cls'](w, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True, choices=list(DETECTORS))
    ap.add_argument('--mode', default='agcm', choices=['full', 'agcm', 'sahi'])
    ap.add_argument('--weights', default=None,
                    help='override detector weight path')
    ap.add_argument('--gate', default=None,
                    help='override gate weight path')
    ap.add_argument('--conf', type=float, default=None)
    ap.add_argument('--imsz', type=int, default=0)
    ap.add_argument('--img-dir', default=str(DATA_ROOT / 'images/val'))
    ap.add_argument('--coco', default=str(DATA_ROOT / 'annotations/instances_val.json'))
    ap.add_argument('--protocol', choices=list(PROTOCOLS), default='default')
    ap.add_argument('--thr', type=float, default=0.5, help='gate threshold')
    ap.add_argument('--tile', type=int, default=640)
    ap.add_argument('--overlap', type=float, default=None,
                    help='base-grid overlap ratio (default: 0.2 for SAHI, '
                         '0.0 otherwise)')
    ap.add_argument('--ablate', choices=['none', 'gf', 'ccm', 'apo'],
                    default='none',
                    help='ablate one AGCM component (mode=agcm only): '
                         'gf=all tiles accepted, ccm=no merged regions, '
                         'apo=plain tiles')
    ap.add_argument('--apo-mode', choices=['auto', 'neighbor', 'cut', 'mix'],
                    default='auto',
                    help='APO expansion driver (auto: hybrid cut|neighbour '
                         'when a multi-cut gate is loaded)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--nms-iou', type=float, default=None)
    ap.add_argument('--postprocess', choices=['nms', 'nmm'], default='nms')
    ap.add_argument('--fuse-large', action='store_true',
                    help='cross-crop fusion for large boxes (sparse scenes)')
    ap.add_argument('--fuse-mode', choices=['wbf', 'union', 'cut'], default='cut')
    ap.add_argument('--fuse-area', type=float, default=20000.0)
    ap.add_argument('--predecode', action='store_true',
                    help='decode all images to RAM first (excludes I/O)')
    ap.add_argument('--out', default='results/agcm_eval.json')
    args = ap.parse_args()
    if args.overlap is None:
        args.overlap = 0.2 if args.mode == 'sahi' else 0.0

    import torch
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    cat_ids, n_cls = PROTOCOLS[args.protocol]
    # model classes are 0-based; GT category ids may start at 1 (COCO) or 0
    cls_offset = min(cat_ids) if cat_ids else 0

    img_dir = Path(args.img_dir)
    coco_path = str(args.coco)

    # dataset dict: id -> file name (read from the COCO annotation file)
    import json as _json
    with open(coco_path) as f:
        ann = _json.load(f)
    id2file = {im['id']: im['file_name'] for im in ann['images']}
    ids = sorted(id2file)
    if args.limit:
        ids = ids[:args.limit]
    print(f'eval {len(ids)} imgs | protocol={args.protocol} '
          f'catIds={cat_ids} | mode={args.mode} | overlap={args.overlap} '
          f'| ablate={args.ablate} | apo_mode={args.apo_mode}', flush=True)

    # ------------------------------------------------------------------
    cfg = AGCMConfig()
    cfg.thr = args.thr
    cfg.tile = args.tile
    cfg.overlap = args.overlap
    cfg.n_cls = n_cls
    cfg.cat_ids = cat_ids
    cfg.postprocess = args.postprocess
    cfg.apo_mode = args.apo_mode
    if args.nms_iou is not None:
        cfg.nms_iou = args.nms_iou
    cfg.fuse_large = args.fuse_large
    cfg.fuse_mode = args.fuse_mode
    cfg.fuse_area = args.fuse_area

    det = build_det(args.det, args.imsz, args.conf, device,
                    weights=args.weights)
    if getattr(det, 'conf', None) is not None and det.conf <= 0.01 and (
            cfg.postprocess == 'nmm'
            or (cfg.fuse_large and cfg.fuse_mode == 'union')):
        print('WARNING: union merging (GREEDYNMM) at conf<=0.01 collapses '
              'mAP: low-confidence boxes union into high-confidence tight '
              'boxes (observed SDS mAP75 0.55 -> ~0.10 at conf=0.001). '
              'Keep postprocess=nms unless you re-tuned the protocol.',
              flush=True)
    gate = None
    if args.mode in ('agcm', 'sahi'):
        gw = args.gate or (MODELS_ROOT / 'gate' / 'gate-customtiny-224.pth')
        gate = GateFilter(str(gw), cfg, device=device)

    # ------------------------------------------------------------------
    img_cache = None
    if args.predecode:
        t0 = time.time()
        img_cache = {iid: np.ascontiguousarray(cv2.imread(str(img_dir / fn)))
                     for iid, fn in id2file.items() if iid in ids}
        print(f'predecoded {len(img_cache)} imgs in {time.time()-t0:.0f}s', flush=True)

    preds, stage_ms = [], {}
    t0 = time.time()
    for i, img_id in enumerate(ids):
        if img_cache is not None:
            img = img_cache[img_id]
        else:
            img = cv2.imread(str(img_dir / id2file[img_id]))
            if img is None:
                print(f'!! cannot read {img_id} ({id2file[img_id]}), skip')
                continue
        H, W = img.shape[:2]

        timer = {k: 0.0 for k in cfg.stages}
        if args.mode == 'full':
            boxes = det.detect_jobs_batch([(img, 0, 0)])
            boxes = (boxes if not args.fuse_large
                     else _fuse(boxes, cfg))
        else:
            if args.mode == 'agcm':
                use_gf = args.ablate != 'gf'
                use_ccm = args.ablate != 'ccm'
                use_apo = args.ablate != 'apo'
            else:
                use_gf = use_ccm = use_apo = False
            pipe = AGCMPipeline(gate, det, cfg, use_gf=use_gf,
                                use_ccm=use_ccm, use_apo=use_apo,
                                overlap=args.overlap,
                                full_img=(args.mode == 'sahi'))
            boxes = pipe(img, timer=timer, gate_thr=args.thr)

        for b in boxes:
            preds.append({'image_id': img_id,
                          'category_id': int(b['class']) + cls_offset,
                          'bbox': [round(v, 2) for v in b['bbox']],
                          'score': round(b['confidence'], 4)})
        for k, v in timer.items():
            stage_ms[k] = stage_ms.get(k, 0.0) + v
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(ids)}] preds={len(preds)}', flush=True)
    total = time.time() - t0

    # ------------------------------------------------------------------
    met = evaluate_coco(preds, coco_path, cat_ids=cat_ids,
                        max_dets=(100, 300, 1000),
                        subset_ids=ids)
    result = {'det': args.det, 'mode': args.mode,
              'n_imgs': len(ids), 'n_pred': len(preds),
              'total_s': round(total, 2),
              'img/s': round(len(ids) / total, 2),
              'ms/img': round(1000 * total / len(ids), 2),
              'stage_ms': {k: round(v / max(len(ids), 1), 1)
                           for k, v in stage_ms.items()},
              'mAP': met.asdict()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def _fuse(boxes, cfg):
    from agcm.pipeline import (nms_boxes, nmm_boxes, cut_fuse_boxes,
                               fuse_large_boxes)
    if cfg.fuse_mode == 'cut':
        return cut_fuse_boxes(boxes, cfg.nms_iou, cfg.n_cls,
                              min_area=cfg.fuse_area)
    if cfg.fuse_mode == 'union':
        return nmm_boxes(boxes, cfg.nms_iou, cfg.n_cls)
    return fuse_large_boxes(boxes, cfg.nms_iou, cfg.n_cls,
                            center_max=cfg.fuse_center,
                            strategy=cfg.fuse_strategy)


if __name__ == '__main__':
    main()
