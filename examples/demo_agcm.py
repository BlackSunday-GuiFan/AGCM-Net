"""End-to-end demo: run AGCM-Net on a single high-resolution frame.

Prereq: model weights available (see models/README.md) and the AGCM_DATA_ROOT
env var pointing at a dataset with images/val. This script does not require
GT annotations -- it just draws the detected boxes.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agcm import AGCMConfig, GateFilter, AGCMPipeline
from agcm.config import DATA_ROOT, MODELS_ROOT
from agcm.detectors import YOLODetector

CLASS_NAMES = ['swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy']
COLORS = [(220, 20, 60), (0, 200, 0), (0, 0, 255), (255, 140, 0), (128, 0, 128)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image', help='path to an image file (any resolution)')
    ap.add_argument('--gate', default=None,
                    help='gate weight (default: AGCM_MODELS_ROOT/gate/...)')
    ap.add_argument('--det', default=None,
                    help='detector weight (default: AGCM_MODELS_ROOT/yolo12n/...)')
    ap.add_argument('--tile', type=int, default=640)
    ap.add_argument('--thr', type=float, default=0.5, help='gate threshold')
    ap.add_argument('--conf', type=float, default=0.25, help='det conf for draw')
    ap.add_argument('--out', default='agcm_demo_out.jpg')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f'cannot read {args.image}')

    cfg = AGCMConfig()
    cfg.tile = args.tile
    cfg.thr = args.thr

    gw = args.gate or str(MODELS_ROOT / 'gate' / 'gate-customtiny-224.pth')
    dw = args.det or str(MODELS_ROOT / 'yolo12n' / 'yolo12n-tile-640.pt')
    gate = GateFilter(gw, cfg, device=args.device)
    det = YOLODetector(dw, imsz=cfg.det_imsz, conf=cfg.conf,
                       device=args.device)

    pipe = AGCMPipeline(gate, det, cfg)
    boxes = pipe(img)
    print(f'{len(boxes)} detections on {img.shape[1]}x{img.shape[0]} image')

    for b in boxes:
        x, y, w, h = [int(v) for v in b['bbox']]
        cls, conf = b['class'], b['confidence']
        if conf < args.conf:
            continue
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f'{CLASS_NAMES[cls]} {conf:.2f}'
        cv2.putText(img, label, (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(args.out, img)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
