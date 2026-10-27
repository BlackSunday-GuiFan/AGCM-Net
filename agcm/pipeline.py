"""AGCM-Net inference pipeline (component-composable, 表2 semantics).

Cascade per row (all tile rows use the yolo12n_tile detector):

  SAHI         : grid(20% overlap) -> every tile AND the full image -> NMS
  Ours (Full)  : grid(0) -> GF accept -> CCM merged regions (large targets)
                 + EVERY accepted tile through APO (expand toward accepted
                 neighbor side(s), 20%) -> batch detect -> NMS
  Ours w/o GF  : grid(0) -> all tiles accepted (GF off) -> CCM collapses to
                 full-image region + every tile APO-expands ALL sides (20%)
                 -> effectively SAHI-like, batch detect -> NMS
  Ours w/o CCM : grid(0) -> GF accept -> NO merged regions; every accepted
                 tile through APO -> batch detect -> NMS
  Ours w/o APO : grid(0) -> GF accept -> CCM merged regions + every accepted
                 tile PLAIN (640, no expansion) -> batch detect -> NMS

APO direction rule (user-confirmed D1): a tile expands toward side(s) where
the NEIGHBORING GRID TILE was GF-accepted (signal side). Merged CCM regions
and every accepted tile are ALL sent to batch detection (not either/or).

cfg.apo_mode selects the expansion driver: 'neighbor' (D1 rule; the behaviour
of has-only gates), 'cut' (multi-cut gate cut heads only), 'mix' (cut head
fires OR neighbour accepted), 'auto' (= mix when a multi-cut gate is loaded,
else neighbor).
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .config import AGCMConfig
from .gate import GateFilter
from .ccm import ConnectedComponentMerger
from .overlay import plain_grid
from .detectors import Detector


def nms_boxes(boxes, iou_thr: float = 0.5, n_cls: int = 5):
    """Per-class greedy NMS on dict boxes (min-area reference, as in paper).

    Vectorized with numpy (per-class suppression matrix + greedy keep):
    identical min-area semantics (inter / min(area1, area2) > iou_thr) to
    the original nested Python loop, ~50x faster at N~300 (video streams).
    """
    if len(boxes) <= 1:
        return boxes
    x1 = np.array([b['bbox'][0] for b in boxes], dtype=np.float64)
    y1 = np.array([b['bbox'][1] for b in boxes], dtype=np.float64)
    x2 = x1 + np.array([b['bbox'][2] for b in boxes], dtype=np.float64)
    y2 = y1 + np.array([b['bbox'][3] for b in boxes], dtype=np.float64)
    sc = np.array([b['confidence'] for b in boxes], dtype=np.float64)
    cl = np.array([b['class'] for b in boxes], dtype=np.int64)

    area = (x2 - x1) * (y2 - y1)
    keep_all = np.zeros(len(boxes), dtype=bool)

    for c in range(n_cls):
        idx = np.where(cl == c)[0]
        if len(idx) == 0:
            continue
        # order by confidence descending (stable)
        order = idx[np.argsort(-sc[idx], kind='stable')]
        k = len(order)
        if k == 0:
            continue
        xx1 = x1[order][:, None]; yy1 = y1[order][:, None]
        xx2 = x2[order][:, None]; yy2 = y2[order][:, None]
        ix1 = np.maximum(xx1, xx1.T); iy1 = np.maximum(yy1, yy1.T)
        ix2 = np.minimum(xx2, xx2.T); iy2 = np.minimum(yy2, yy2.T)
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        ovr = inter / np.minimum(area[order][:, None], area[order][None, :])
        # greedy: iterate in confidence order; suppress only LATER boxes
        # (matches original nested loop semantics: i2 in range(i1+1, ...))
        discarded = np.zeros(k, dtype=bool)
        for i in range(k):
            if discarded[i]:
                continue
            mask = ovr[i] > iou_thr
            mask[:i + 1] = False          # keep self and earlier kept boxes
            discarded = discarded | mask
        keep_all[order[~discarded]] = True

    return [boxes[i] for i in range(len(boxes)) if keep_all[i]]


def nmm_boxes(boxes, iou_thr: float = 0.5, n_cls: int = 5):
    """Per-class greedy Non-Maximum Merging (SAHI GREEDYNMM semantics):
    overlapping boxes are UNION-merged instead of suppressed, so partial
    boxes of a large object seen by several tiles fuse back into the full
    object. Confidence of a merged box = max of its members."""
    if len(boxes) <= 1:
        return boxes
    out = []
    for c in range(n_cls):
        cls_boxes = [b for b in boxes if b['class'] == c]
        if not cls_boxes:
            continue
        cls_boxes.sort(key=lambda b: -b['confidence'])
        merged = []          # list of dict boxes (unions)
        for b in cls_boxes:
            bx, by, bw, bh = b['bbox']
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            best_i = None
            for i, m in enumerate(merged):
                mx, my, mw, mh = m['bbox']
                mx1, my1, mx2, my2 = mx, my, mx + mw, my + mh
                ix1, iy1 = max(x1, mx1), max(y1, my1)
                ix2, iy2 = min(x2, mx2), min(y2, my2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                a1 = (x2 - x1) * (y2 - y1)
                a2 = (mx2 - mx1) * (my2 - my1)
                ovr = inter / max(min(a1, a2), 1e-9)
                if ovr > iou_thr:
                    best_i = i
                    break
            if best_i is not None:
                m = merged[best_i]
                mx, my, mw, mh = m['bbox']
                ux1 = min(x1, mx); uy1 = min(y1, my)
                ux2 = max(x2, mx + mw); uy2 = max(y2, my + mh)
                m['bbox'] = [ux1, uy1, ux2 - ux1, uy2 - uy1]
                m['confidence'] = max(m['confidence'], b['confidence'])
            else:
                merged.append(dict(b))
        out.extend(merged)
    return out


def nmm_large_boxes(boxes, iou_thr: float = 0.5, min_area: float = 20000.0,
                    n_cls: int = 5):
    """Hybrid postprocess for sparse large-object scenes (MARIS):
    boxes with area >= min_area are union-merged (GREEDYNMM) so truncated
    cross-tile detections of one large object fuse back into the full box
    (a truncated APO box can otherwise outscore the full CCM box and win
    plain NMS); small boxes keep the standard min-area NMS and are dropped
    when covered by a merged large box."""
    if len(boxes) <= 1:
        return boxes
    out = []
    for c in range(n_cls):
        cand = [b for b in boxes if b['class'] == c]
        if not cand:
            continue
        large = [b for b in cand if b['bbox'][2] * b['bbox'][3] >= min_area]
        small = [b for b in cand if b['bbox'][2] * b['bbox'][3] < min_area]
        merged = nmm_boxes(large, iou_thr, n_cls) if large else []
        kept = nms_boxes(small, iou_thr, n_cls) if small else []
        for s in kept:
            # drop a small box whose area is mostly covered by a merged
            # large box (min-area IoU: containment, not plain IoU, so a
            # small fragment just below fuse_area still collapses into it)
            if any(_min_area_iou(s['bbox'], m['bbox']) > iou_thr
                   for m in merged):
                continue
            out.append(s)
        out.extend(merged)
    return out


def _min_area_iou(a, b):
    """Overlap ratio inter / min(area_a, area_b): 1.0 when one box fully
    contains the other, unlike plain IoU which caps near area_b/area_a."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    den = min(aw * ah, bw * bh)
    return inter / den if den > 0 else 0.0


def _overlap_span(a, b, axis):
    """1D overlap (px) of spans along the OTHER axis; bbox = [x, y, w, h].
    axis='x' pairs compare y spans (left/right neighbours); axis='y' pairs
    compare x spans (up/down neighbours)."""
    if axis == 'x':                     # compare y spans
        a1, a2 = a[1], a[1] + a[3]
        b1, b2 = b[1], b[1] + b[3]
    else:                               # compare x spans
        a1, a2 = a[0], a[0] + a[2]
        b1, b2 = b[0], b[0] + b[2]
    return max(0, min(a2, b2) - max(a1, b1))


def cut_fuse_boxes(boxes, iou_thr: float = 0.5, n_cls: int = 5,
                   deltar: float = 0.10, min_area: float = 0.0,
                   conf_thr: float = 0.25):
    """Truncation-aware fusion (CA-NMS directional-complement evidence):

    A tile-source box is CONFIRMED TRUNCATED only when it touches its crop
    border AND the neighbouring tile holds a same-class box touching the
    opposite border of that side with overlapping span. Truncated boxes are
    union-merged along these confirmed cut sides into full boxes; complete
    boxes are left untouched. Everything then competes by min-area NMS.

    Sparse large-object scenes (MARIS) recover cross-tile full boxes
    (+AP); dense scenes (SDS) are untouched because in-tile detections do
    not touch crop borders -> zero drift from the NMS baseline.
    min_area > 0 restricts fusion to large boxes (fuse_area semantics).
    Only boxes with confidence >= conf_thr participate in fusion: the
    detector at conf=0.001 emits many near-zero fragments along tile
    borders; fusing them into a real box inflates it far beyond the
    object (SDS -1.4pt). Real truncated boxes score well above 0.25."""
    if len(boxes) <= 1:
        return boxes
    tile_by_tile = defaultdict(list)
    for b in boxes:
        m = b.get('_meta')
        if (m is not None and m.get('kind') == 'tile'
                and b['confidence'] >= conf_thr):
            tile_by_tile[(m['tx'], m['ty'])].append(b)

    def ob_touch(ob, side):
        """Neighbour box touches/crosses its OWN tile border on `side`.
        Tile borders, not crop borders: an APO-expanded crop covers parts of
        neighbour tiles, so the truncation evidence must live on the grid
        line the two tiles share, not on the (expanded) crop rectangle."""
        m = ob['_meta']
        ntx, nty = m['tx'], m['ty']
        ntw, nth = m['tw'], m['th']
        obx, oby, obw, obh = ob['bbox']
        od = deltar * min(obw, obh)
        if side == 'l':
            return obx <= ntx + od and obx + obw > ntx
        if side == 'r':
            return obx + obw >= ntx + ntw - od and obx < ntx + ntw
        if side == 'u':
            return oby <= nty + od and oby + obh > nty
        return oby + obh >= nty + nth - od and oby < nty + nth

    def confirmed_cuts(b):
        """directions where b has complementary evidence in the neighbour
        tile; also returns the paired neighbour box for union-merge.
        A tile border that coincides with the image edge never counts (an
        edge tile is not 'truncated', the target simply ends there)."""
        m = b['_meta']
        tx, ty = m['tx'], m['ty']
        tw, th = m['tw'], m['th']
        W, H = m['W'], m['H']
        bx, by, bw, bh = b['bbox']
        delta = deltar * min(bw, bh)
        right, bottom = tx + tw, ty + th
        cand = []
        if right < W - 1e-6 and bx + bw >= right - delta and bx < right:
            cand.append(('r', (tx + tw, ty), 'x'))
        if tx > 1e-6 and bx <= tx + delta and bx + bw > tx:
            cand.append(('l', (tx - tw, ty), 'x'))
        if bottom < H - 1e-6 and by + bh >= bottom - delta and by < bottom:
            cand.append(('d', (tx, ty + th), 'y'))
        if ty > 1e-6 and by <= ty + delta and by + bh > ty:
            cand.append(('u', (tx, ty - th), 'y'))
        opp = {'l': 'r', 'r': 'l', 'u': 'd', 'd': 'u'}
        out = []
        for side, (ntx, nty), axis in cand:
            for ob in tile_by_tile.get((ntx, nty), []):
                if ob['class'] != b['class']:
                    continue
                if _overlap_span(b['bbox'], ob['bbox'], axis) <= 0:
                    continue
                if ob_touch(ob, opp[side]):
                    out.append(ob)
                    break
        return out

    # ---- per-class: union-find truncated boxes, keep complete ones ----
    for c in range(n_cls):
        cand = [b for b in boxes
                if b['class'] == c and b['confidence'] >= conf_thr]
        if not cand:
            continue
        cuts = {}
        for i, b in enumerate(cand):
            m = b.get('_meta')
            if m is None or m.get('kind') != 'tile':
                continue
            if m['x2'] - m['x1'] < m['tw'] * 0.6:      # APO crop far below a
                continue                                # tile: not truncated
            cuts[i] = confirmed_cuts(b)

        parent = {i: i for i in range(len(cand))}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def unite(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        id2idx = {id(b): i for i, b in enumerate(cand)}
        for i, nbs in cuts.items():
            for ob in nbs:
                j = id2idx.get(id(ob))
                if j is not None:
                    unite(i, j)

        groups = defaultdict(list)
        for i in range(len(cand)):
            groups[find(i)].append(i)
        for g, idxs in groups.items():
            if len(idxs) < 2 or not any(i in cuts for i in idxs):
                continue                    # singletons & untouched groups
            mem = [cand[i] for i in idxs]
            area = [b['bbox'][2] * b['bbox'][3] for b in mem]
            if min_area > 0 and max(area) < min_area:
                continue                    # too small for fusion
            bx1 = min(b['bbox'][0] for b in mem)
            by1 = min(b['bbox'][1] for b in mem)
            bx2 = max(b['bbox'][0] + b['bbox'][2] for b in mem)
            by2 = max(b['bbox'][1] + b['bbox'][3] for b in mem)
            merged = dict(mem[0])
            merged['bbox'] = [bx1, by1, bx2 - bx1, by2 - by1]
            merged['confidence'] = max(b['confidence'] for b in mem)
            for b in mem:                   # drop members, append union
                try:
                    boxes.remove(b)
                except ValueError:
                    pass
            boxes.append(merged)
    return nms_boxes(boxes, iou_thr, n_cls)


def _touches(acc_set, tx, ty, tw, th, side):
    """8-neighbour test with fixed ±tile offsets (table-1 semantics): a
    diagonal neighbour accepted also triggers both adjacent sides (e.g.
    top-right accepted -> expand right AND up), so a target cut in a tile
    corner gets fully covered. Exact offsets exclude the tile itself."""
    if side == 'r':
        return ((tx + tw, ty) in acc_set or (tx + tw, ty - th) in acc_set
                or (tx + tw, ty + th) in acc_set)
    if side == 'l':
        return ((tx - tw, ty) in acc_set or (tx - tw, ty - th) in acc_set
                or (tx - tw, ty + th) in acc_set)
    if side == 'd':
        return ((tx, ty + th) in acc_set or (tx - tw, ty + th) in acc_set
                or (tx + tw, ty + th) in acc_set)
    if side == 'u':
        return ((tx, ty - th) in acc_set or (tx - tw, ty - th) in acc_set
                or (tx + tw, ty - th) in acc_set)
    return False


def _iou_xywh(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def _wbf_fuse(boxes, iou_thr: float, n_cls: int, center_max: float,
              strategy: str = 'conf'):
    """Fusion of large-box clusters: greedy clusters in confidence order.
    strategy 'conf'  : confidence-weighted mean of members (classic WBF)
    strategy 'area'  : area-weighted mean (the full/CCM box, being larger,
                       dominates, so partial tiles cannot shrink it)
    strategy 'maxbox': keep the largest member box (recover-only, never
                       shrinks; milder than union, no averaging)
    Boxes merge only when overlapping AND centers close (<= center_max px),
    so repeated detections of one object across adjacent APO/CCM crops
    fuse while distinct nearby objects do not."""
    def cdist(a, b):
        return max(abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)),
                   abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)))
    out = []
    for c in range(n_cls):
        cand = [b for b in boxes if b['class'] == c]
        cand.sort(key=lambda b: -b['confidence'])
        clusters = []
        for b in cand:
            bb = b['bbox']
            for cl in clusters:
                if (_iou_xywh(cl['box'], bb) >= iou_thr
                        and cdist(cl['box'], bb) <= center_max):
                    s = cl['sum_conf'] + b['confidence']
                    if strategy == 'maxbox':
                        if bb[2] * bb[3] > cl['box'][2] * cl['box'][3]:
                            cl['box'] = list(bb)
                    elif strategy == 'area':
                        wa = cl['area'] + bb[2] * bb[3]
                        w = (bb[2] * bb[3]) / wa
                        cl['box'] = [cl['area'] / wa * v + w * nv
                                     for v, nv in zip(cl['box'], bb)]
                        cl['area'] = wa
                    else:                       # 'conf' classic WBF
                        w = b['confidence'] / s
                        cl['box'] = [cl['sum_conf'] / s * v + w * nv
                                     for v, nv in zip(cl['box'], bb)]
                    cl['sum_conf'] = s
                    cl['conf'] = max(cl['conf'], b['confidence'])
                    break
            else:
                clusters.append({'box': list(bb),
                                 'sum_conf': b['confidence'],
                                 'area': bb[2] * bb[3],
                                 'conf': b['confidence']})
        for cl in clusters:
            out.append({'bbox': cl['box'], 'confidence': cl['conf'],
                        'class': c})
    return out


def fuse_large_boxes(boxes, iou_thr: float = 0.5, min_area: float = 20000.0,
                     n_cls: int = 5, center_max: float = 256.0,
                     strategy: str = 'conf'):
    """Cross-crop box fusion for LARGE boxes + greedy NMS for the rest.

    Validated on MARIS (maris val, AGCM tile-apo): AP_all 0.5956 -> 0.5970
    (+1.4), every size bucket unchanged or better, cost ~0. Large objects
    (area >= min_area px²) are detected repeatedly across adjacent
    APO-expanded / CCM-merged crops; weighted fusion averages them into a
    tighter box. Small boxes keep the standard min-area greedy NMS.
    """
    if not boxes:
        return []
    large = [b for b in boxes if b['bbox'][2] * b['bbox'][3] >= min_area]
    small = [b for b in boxes if b['bbox'][2] * b['bbox'][3] < min_area]
    return _wbf_fuse(large, iou_thr, n_cls, center_max, strategy) \
        + nms_boxes(small, iou_thr, n_cls)


class AGCMPipeline:
    """Composable AGCM cascade: grid -> [GF] -> [CCM] -> [APO] -> detect -> NMS."""

    def __init__(self, gate, detector: Detector, cfg: AGCMConfig,
                 use_gf=True, use_ccm=True, use_apo=True,
                 overlap: float = 0.0, full_img: bool = False,
                 standard_det=None, device='cuda:0'):
        self.gate = gate
        self.det = detector
        self.cfg = cfg
        self.use_gf = use_gf
        self.use_ccm = use_ccm
        self.use_apo = use_apo
        self.overlap = overlap
        self.full_img = full_img       # also detect the whole image (SAHI)
        self.standard_det = standard_det  # optional higher-res detector for
                                          # the full-image pass (SAHI std pred)
        self.ccm = ConnectedComponentMerger(cfg)

    # ------------------------------------------------------------------
    def _apo_crop(self, img, tx, ty, acc_set, H, W, tw=None, th=None):
        """APO expansion of tile (tx,ty) toward sides whose NEIGHBORING GRID
        TILE is GF-accepted. 20% one-sided expansion on signal side(s).

        8-neighbour rule: a diagonal neighbour accepted also triggers both
        adjacent sides (e.g. top-right accepted -> expand right AND up), so
        a target cut in a tile corner gets fully covered.
        """
        tw = tw or self.cfg.tile
        th = th or self.cfg.tile
        ex_w = int(round(tw * self.cfg.expand))
        ex_h = int(round(th * self.cfg.expand))
        has_r = _touches(acc_set, tx, ty, tw, th, 'r')
        has_d = _touches(acc_set, tx, ty, tw, th, 'd')
        has_l = _touches(acc_set, tx, ty, tw, th, 'l')
        has_u = _touches(acc_set, tx, ty, tw, th, 'u')
        if not (has_r or has_d or has_l or has_u):
            return (img[ty:min(ty + th, H), tx:min(tx + tw, W)], tx, ty)
        x1 = max(0, tx - ex_w) if has_l else tx
        y1 = max(0, ty - ex_h) if has_u else ty
        x2 = min(tx + tw + ex_w, W) if has_r else tx + tw
        y2 = min(ty + th + ex_h, H) if has_d else ty + th
        return (img[y1:y2, x1:x2], x1, y1)

    def _apo_crop_cut(self, img, tx, ty, cuts, H, W, tw=None, th=None):
        """APO expansion driven by the multi-cut gate: expand exactly the
        border(s) the gate says truncate a target (cut_l->left, cut_r->right,
        cut_t->up, cut_b->down). Falls back to the plain tile when no border
        is flagged."""
        tw = tw or self.cfg.tile
        th = th or self.cfg.tile
        ex_w = int(round(tw * self.cfg.expand))
        ex_h = int(round(th * self.cfg.expand))
        c = cuts.get((tx, ty), [0.0, 0.0, 0.0, 0.0])
        thr = self.cfg.cut_thr
        x1 = max(0, tx - ex_w) if c[0] >= thr else tx
        x2 = min(tx + tw + ex_w, W) if c[1] >= thr else tx + tw
        y1 = max(0, ty - ex_h) if c[2] >= thr else ty
        y2 = min(ty + th + ex_h, H) if c[3] >= thr else ty + th
        if (x1, y1, x2, y2) == (tx, ty, tx + tw, ty + th):
            return img[ty:min(ty + th, H), tx:min(tx + tw, W)], tx, ty
        return img[y1:y2, x1:x2], x1, y1

    def _apo_crop_mix(self, img, tx, ty, cuts, acc_set, H, W, tw=None,
                      th=None):
        """APO expansion = cut head fires OR the neighbouring grid tile is
        GF-accepted (hybrid: cut-driven precision with neighbour-rule
        recall, so small targets whose cut head misses still get covered)."""
        tw = tw or self.cfg.tile
        th = th or self.cfg.tile
        ex_w = int(round(tw * self.cfg.expand))
        ex_h = int(round(th * self.cfg.expand))
        c = cuts.get((tx, ty), [0.0, 0.0, 0.0, 0.0])
        thr = self.cfg.cut_thr
        x1 = max(0, tx - ex_w) if (c[0] >= thr or
                                   _touches(acc_set, tx, ty, tw, th,
                                            'l')) else tx
        x2 = min(tx + tw + ex_w, W) if (c[1] >= thr or
                                        _touches(acc_set, tx, ty, tw,
                                                 th, 'r')) else tx + tw
        y1 = max(0, ty - ex_h) if (c[2] >= thr or
                                   _touches(acc_set, tx, ty, tw, th,
                                            'u')) else ty
        y2 = min(ty + th + ex_h, H) if (c[3] >= thr or
                                        _touches(acc_set, tx, ty, tw,
                                                 th, 'd')) else ty + th
        if (x1, y1, x2, y2) == (tx, ty, tx + tw, ty + th):
            return img[ty:min(ty + th, H), tx:min(tx + tw, W)], tx, ty
        return img[y1:y2, x1:x2], x1, y1

    def _tile_crop(self, img, tx, ty, H, W, tw=None, th=None):
        tw = tw or self.cfg.tile
        th = th or self.cfg.tile
        return (img[ty:min(ty + th, H), tx:min(tx + tw, W)], tx, ty)

    # ------------------------------------------------------------------
    def _run(self, img, timer=None, gate_thr=None):
        H, W = img.shape[:2]
        if self.cfg.tile_adaptive and self.overlap == 0:
            from .overlay import adaptive_tile, int_grid
            tw = adaptive_tile(W, self.cfg.tile, self.cfg.tile_lo,
                               self.cfg.tile_hi)
            th = adaptive_tile(H, self.cfg.tile, self.cfg.tile_lo,
                               self.cfg.tile_hi)
            tiles = int_grid(W, H, tw, th)
        else:
            tw = th = self.cfg.tile
            tiles = plain_grid(W, H, tw, self.overlap)
        self._tw = tw
        self._th = th
        if not tiles:
            return []

        jobs = []

        # ---- GF: accept or accept-all ----
        if self.use_gf:
            accepted, _, _ = self.gate.filter_tiles(
                img, tiles, fast=True, thr=gate_thr, timer=timer,
                tile_hw=(self._tw, self._th))
        else:
            accepted = tiles                          # GF off: all accepted
        acc_set = set(accepted)
        if not accepted:
            return []

        # ---- CCM: merged regions (large targets crossing tile borders) ----
        if self.use_ccm:
            t_cc = time.time()
            comps = self.ccm.components(accepted, self._tw, self._th)
            for comp in comps:
                if len(comp) > 1:
                    x1, y1, x2, y2 = self.ccm.merged_rect(
                        comp, H, W, self._tw, self._th)
                    jobs.append((img[y1:y2, x1:x2], x1, y1,
                                 {'kind': 'ccm', 'x1': x1, 'y1': y1,
                                  'x2': x2, 'y2': y2, 'tw': self._tw,
                                  'th': self._th, 'W': W, 'H': H}))
            if timer is not None:
                timer['cc'] = time.time() - t_cc

        # ---- every accepted tile -> APO expansion or plain ----
        apo_mode = self.cfg.apo_mode
        if apo_mode == 'auto':      # hybrid expansion with multi-cut gates
            apo_mode = 'mix' if getattr(self.gate, 'multi_cut', False) \
                else 'neighbor'
        cuts = getattr(self.gate, 'cuts', None) or {}
        for tx, ty in accepted:
            if self.use_apo:
                if apo_mode == 'mix':
                    crop, cx1, cy1 = self._apo_crop_mix(
                        img, tx, ty, cuts, acc_set, H, W,
                        self._tw, self._th)
                elif apo_mode == 'cut':
                    crop, cx1, cy1 = self._apo_crop_cut(
                        img, tx, ty, cuts, H, W, self._tw, self._th)
                else:
                    crop, cx1, cy1 = self._apo_crop(
                        img, tx, ty, acc_set, H, W, self._tw, self._th)
            else:
                crop, cx1, cy1 = self._tile_crop(img, tx, ty, H, W,
                                                 self._tw, self._th)
            jobs.append((crop, cx1, cy1,
                         {'kind': 'tile', 'tx': tx, 'ty': ty,
                          'x1': cx1, 'y1': cy1,
                          'x2': cx1 + crop.shape[1],
                          'y2': cy1 + crop.shape[0],
                          'tw': self._tw, 'th': self._th,
                          'W': W, 'H': H}))

        # ---- SAHI variant: the whole image is also detected ----
        full_box = None
        if self.full_img:
            if self.standard_det is not None:
                # higher-resolution standard prediction (e.g. 1280) run on
                # the full image separately, then merged with tile detections
                full_box = self.standard_det.detect_jobs_batch([(img, 0, 0)])
            else:
                jobs.append((img, 0, 0))

        # ---- batch detect + NMS (large-box cross-crop fusion) ----
        t_det = time.time()
        boxes = self.det.detect_jobs_batch(jobs)
        if full_box is not None:
            boxes = list(boxes) + list(full_box)
        if timer is not None:
            timer['detect'] = time.time() - t_det
        t0 = time.time()
        if self.cfg.fuse_large:
            if self.cfg.fuse_mode == 'union':
                boxes = nmm_large_boxes(boxes, self.cfg.nms_iou,
                                        self.cfg.fuse_area,
                                        n_cls=self.cfg.n_cls)
            elif self.cfg.fuse_mode == 'cut':
                boxes = cut_fuse_boxes(boxes, self.cfg.nms_iou,
                                       n_cls=self.cfg.n_cls,
                                       min_area=self.cfg.fuse_area)
            else:
                boxes = fuse_large_boxes(boxes, self.cfg.nms_iou,
                                         self.cfg.fuse_area,
                                         n_cls=self.cfg.n_cls,
                                         center_max=self.cfg.fuse_center,
                                         strategy=self.cfg.fuse_strategy)
        elif self.cfg.postprocess == 'nmm':
            boxes = nmm_boxes(boxes, self.cfg.nms_iou, self.cfg.n_cls)
        else:
            boxes = nms_boxes(boxes, self.cfg.nms_iou, self.cfg.n_cls)
        if timer is not None:
            timer['nms'] = time.time() - t0
        return boxes

    # ------------------------------------------------------------------
    def __call__(self, img, timer=None, gate_thr=None):
        return self._run(img, timer, gate_thr)


def run_agcm_image(img, img_id, gate, det, cfg, *, use_gf=True,
                   use_ccm=True, use_apo=True, overlap=0.0,
                   gate_thr=0.5, timer=None):
    """Convenience: run pipeline -> COCO-format preds (1-based categories)."""
    pipe = AGCMPipeline(gate, det, cfg, use_gf=use_gf, use_ccm=use_ccm,
                        use_apo=use_apo, overlap=overlap)
    boxes = pipe(img, timer=timer, gate_thr=gate_thr)
    return [{'image_id': img_id, 'category_id': int(b['class']) + 1,
             'bbox': [round(v, 2) for v in b['bbox']],
             'score': round(b['confidence'], 4)} for b in boxes]