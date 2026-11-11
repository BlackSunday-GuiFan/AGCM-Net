# AGCM-Net

**Merge-then-Detect Inference for High-Resolution Aerial Maritime Imagery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)]()

Reference implementation of the framework described in *AGCM-Net: Merge-then-Detect Inference for High-Resolution Aerial Maritime Imagery* (Fan Gui, Zhangsong Shi, Weiqiang Hu, Huihui Xu, Miao Zhao).

AGCM-Net is a **detector-agnostic inference framework** for small-object detection in high-resolution (up to 4K) aerial imagery from **civilian maritime search and rescue**. It reorganizes tiled inference into a **merge-then-detect** order: a lightweight gate produces a binary mask over the tile grid, and the connected components of that mask (8-connectivity) identify neighbouring tiles that share an object. One structure therefore both discards empty sea and reassembles cross-boundary objects **before the detector is invoked**. An adaptive one-sided overlap policy and tile-level training complete the framework. This is the opposite of SAHI's detect-then-merge, where an object cut by the grid cannot reliably be rebuilt from partial boxes by post-hoc merging.

On the SeaDronesSee validation set, AGCM-Net raises YOLOv12n from 0.3093 to **0.5635 AP50-95** at **35.4 FPS** on high-resolution frames. Against the strongest tiled baseline it leads in overall AP (0.5635 vs 0.5372), large-object AP (0.6720 vs 0.6282) and geometric completeness (DFR@0.75 0.127 vs 0.207; AIoU<sub>L</sub> 0.873 vs 0.841) at about **2.5 times the throughput**.

---

## Results

### Main comparison — SeaDronesSee validation set

All rows use the same tile-level-trained weights where applicable; `@n` is the input resolution for full-image rows and the tile side for tiled rows. AGCM-Net uses a 640-pixel adaptive grid, gate threshold τ = 0.3, one-sided expansion e = 0.1 and truncation-aware fusion; SAHI uses 640 × 640 tiles with 10% overlap.

| Configuration | AP50-95 | AP50 | AP<sub>S</sub> | AP<sub>L</sub> | DFR@0.75 | AIoU<sub>L</sub> | FPS |
|---|---|---|---|---|---|---|---|
| YOLOv12n @640 (full image) | 0.3093 | 0.5606 | 0.1848 | 0.5039 | 0.174 | 0.838 | 79.0 |
| YOLOv12n @1280 (full image) | 0.4185 | 0.7306 | 0.2817 | 0.6001 | 0.176 | 0.841 | 49.6 |
| SAHI @640 | 0.5372 | 0.8733 | 0.5219 | 0.6282 | 0.207 | 0.841 | 14.3 |
| **AGCM-Net (YOLOv12n) @640** | **0.5635** | **0.8841** | 0.5186 | **0.6720** | **0.127** | **0.873** | **35.4** |

The gains transfer across three heterogeneous lightweight detectors — YOLOv12n, DEIMv2-Nano and PP-PicoDet-S — and to the independent MARIS benchmark (SAHI trail 4.7 AP points; 1.8 times throughput). On the dense VisDrone2019 benchmark the gate accepts 98.9% of tiles, which quantifies the sparsity boundary of the approach: as object density rises, the framework drifts back towards full-image inference.

Two detection-level fragmentation metrics, **DFR@0.75** and **AIoU<sub>L</sub>**, expose a structural limitation of tiled inference that overall AP hides. A single operating point — the tile-level recall of the gate, R<sub>gate</sub> — makes the accuracy-latency trade-off explicit; the default R<sub>gate</sub> = 0.95 gives 0.5652 AP50-95 at 25.2 FPS, and a search-and-rescue workload that cannot afford misses can push it towards 1.00.

### Edge deployment

On an NVIDIA Jetson Orin NX (Ampere, 16 GB, TensorRT 8.5.2), with both the gate and detector converted to FP16 engines and 3840 × 2160 SeaDronesSee validation frames, the full pipeline runs at **77.3 ms per frame (12.9 FPS)** excluding image decoding, with **AP50-95 = 0.5605** — within 0.003 of the workstation result. Roughly 80% of tiles are rejected in sparse maritime scenes, so embedded computation concentrates on the few tiles containing targets.

---

## How it works

| Module | Purpose |
|---|---|
| `GF` (Gated Filtering) | Lightweight gate CNN (`CustomTiny` in `agcm/gate.py`, called TinyVGG in the paper; 0.39 M parameters, 0.23 ms/tile) scores 224 × 224 patches and produces a binary mask over the tile grid |
| `CCM` (Connected-Component Merging) | Groups accepted tiles into regions with **8-connectivity** and builds merged detection crops *before* detection |
| `APO` (Adaptive One-sided Expansion) | Expands an accepted tile only toward sides where a neighbour was accepted, restoring boundary context without paying for uniform overlap |
| `DET` | Detector adapters (Ultralytics YOLO / PaddleDetection PicoDet / mmdetection RTMDet / DEIMv2) behind one interface |
| `Eval` | COCO official protocol, DFR@0.75 / AIoU<sub>L</sub>, throughput and per-stage latency |

```
full image ──► adaptive integer tiling (near 640 px, ±20%)
     │
     ▼
GF: score every tile (TinyVGG, 224×224) ──► binary mask over the grid
     │
     ▼
CCM: 8-connected components of accepted tiles ──► merged regions
     │                                              (cross-boundary objects whole)
     ▼
APO: each accepted tile expands toward accepted neighbours (one-sided)
     │
     ▼
batch detect (accepted tiles + merged regions) ──► global NMS / truncation-aware fusion
```

Because the detector sees the whole object from the start, NMS never has to arbitrate between partial boxes of the same target.

---

## Installation

```bash
git clone https://github.com/BlackSunday-GuiFan/AGCM-Net.git
cd AGCM-Net
pip install -r requirements.txt
pip install -e .
```

Core dependencies: `torch`, `ultralytics`, `numpy`, `opencv-python`, `pycocotools`.

> PicoDet / RTMDet / DEIMv2 adapters require their own frameworks (PaddleDetection / mmdetection / DEIMv2) and are optional. The core pipeline runs with the Ultralytics YOLO adapter alone; the CLI runner registers the YOLO variants.

## Data and model weights

Experiments use three COCO-format datasets:

| Dataset | Val images | Classes | Objects / image | Resolution | Regime |
|---|---|---|---|---|---|
| SeaDronesSee | 1547 | 5 | 6.2 | 3840 × 2160 | sparse maritime |
| MARIS | 364 | 3 | 15.2 | 3840 × 2160 | moderate density |
| VisDrone2019 | 548 | 10 | 70.7 | 1360 × 765 | dense aerial |

SeaDronesSee is publicly available. MARIS was assembled and annotated for this study and will be released upon publication.

```bash
export AGCM_DATA_ROOT=/path/to/dataset   # expects images/val + annotations/instances_val.json
export AGCM_MODELS_ROOT=/path/to/weights # expects gate/ + yolo12n/ (see models/README.md)
```

Checkpoints are **not shipped** in this repository. See [models/README.md](models/README.md) for the expected layout and training procedure.

## Quick start

```python
import cv2
from agcm import AGCMConfig, GateFilter, AGCMPipeline
from agcm.detectors import YOLODetector

cfg = AGCMConfig()                                   # tile=640, gate=224, thr=0.5
gate = GateFilter(cfg.DEFAULTS['gate_weight'], cfg, device='cuda:0')
det = YOLODetector(cfg.DEFAULTS['yolo12n_tile'], imsz=640, device='cuda:0')

img = cv2.imread('/path/to/4k_frame.jpg')            # BGR, any resolution
pipe = AGCMPipeline(gate, det, cfg)
boxes = pipe(img)                                    # [{'bbox':[x,y,w,h], 'class':int, 'confidence':float}]
```

Or run the CLI experiment runner:

```bash
python scripts/run_experiment.py --det yolo12n_tile --mode agcm
python scripts/run_experiment.py --det yolo12n_full --mode full
python scripts/run_experiment.py --det yolo12n_tile --mode sahi
```

`--ablate {gf,ccm,apo}` reproduces the ablation rows, and `--protocol {default,maris,visdrone}` selects the dataset's category-id registry.

### Reproducing the paper's operating point

The paper's main tables use τ = 0.3, one-sided expansion e = 0.1 and truncation-aware fusion:

```bash
python scripts/run_experiment.py --det yolo12n_tile --mode agcm \
    --thr 0.3 --fuse-large --fuse-mode cut
```

The expansion fraction is currently a configuration field (`AGCMConfig.expand`, default 0.2) rather than a CLI flag; set it to 0.1 to match the paper exactly. The CLI defaults (`thr=0.5`, `expand=0.2`, `fuse_large=False`) are the library defaults and do not reproduce the reported numbers as-is.

## Evaluation protocol

- COCO official evaluation: AP50-95, AP50, and area-stratified AP<sub>S</sub> / AP<sub>M</sub> / AP<sub>L</sub> with area bins below 32², 32²-96² and above 96² pixels; `maxDets=[100,300,1000]`.
- Category ids are **dataset-specific** and taken from the protocol registry in `scripts/run_experiment.py`: `[0..4]` for SeaDronesSee V2, `[1..3]` for MARIS, `[1..10]` for VisDrone2019. Detector classes are 0-based; the runner offsets GT ids by `min(cat_ids)` and skips `iscrowd` annotations.
- Detector `conf=0.001`; DFR@0.75 and AIoU<sub>L</sub> are reported alongside COCO AP.
- Throughput is **pure inference** on a single GPU, excluding image decoding, so it isolates the detector and pipeline cost. Use `--predecode` to exclude I/O explicitly.

## Citing

The paper describing this framework is not yet published; please cite this repository until a DOI is available.

```bibtex
@misc{agcmnet2026,
  title        = {AGCM-Net: Merge-then-Detect Inference for High-Resolution Aerial Maritime Imagery},
  author       = {Gui, Fan and Shi, Zhangsong and Hu, Weiqiang and Xu, Huihui and Zhao, Miao},
  year         = {2026},
  howpublished = {\url{https://github.com/BlackSunday-GuiFan/AGCM-Net}},
  note         = {Manuscript; reference implementation}
}
```

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (General Program, Grant No. 42676231; Young Scientists Fund, Grant No. 62506378).

## License

AGCM-Net is released under the [MIT License](LICENSE).
