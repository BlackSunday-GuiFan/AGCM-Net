"""CCM - Connected-Component Merging.

Groups spatially-adjacent accepted grid tiles (4-connectivity) into regions,
then builds detection crops:
  - singleton component  -> the tile crop itself (for small targets)
  - multi-tile component -> ONE merged bounding crop covering the whole
    component (restores targets truncated by grid boundaries) PLUS each
    individual tile crop (retains small-target resolution).

The merged region is the axis-aligned rect of the component's bounding box,
filled/cropped to the detector input size by the caller's crop runner.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from scipy import ndimage

from .config import AGCMConfig

Pos = Tuple[int, int]


class ConnectedComponentMerger:
    """CCM: accepted-tile mask -> connected components -> detection crops."""

    def __init__(self, cfg: AGCMConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def components(self, accepted: List[Pos], tw: int | None = None,
                   th: int | None = None):
        """Return list of components; each component is a list of (tx, ty).

        With tw/th (actual grid step) real-coordinate 4-connectivity is used:
        two accepted tiles merge only when their grid distance is exactly
        one step (tx diff == tw or ty diff == th), so tiles separated by a
        rejected tile are NOT merged. Without tw/th (legacy callers) the
        old sorted-index adjacency is kept.
        """
        if not accepted:
            return []
        if tw is not None and th is not None:
            acc = set(accepted)
            parent = {t: t for t in accepted}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for (tx, ty) in accepted:
                for (nx, ny) in ((tx - tw, ty), (tx + tw, ty),
                                 (tx, ty - th), (tx, ty + th)):
                    if (nx, ny) in acc:
                        union((tx, ty), (nx, ny))
            comps = {}
            for t in accepted:
                comps.setdefault(find(t), []).append(t)
            return list(comps.values())
        cx = sorted({tx for tx, ty in accepted})
        cy = sorted({ty for tx, ty in accepted})
        xi = {v: i for i, v in enumerate(cx)}
        yi = {v: i for i, v in enumerate(cy)}
        mask = np.zeros((len(cy), len(cx)), dtype=bool)
        for tx, ty in accepted:
            mask[yi[ty], xi[tx]] = True
        lab, _ = ndimage.label(mask, structure=np.array([[0, 1, 0],
                                                         [1, 1, 1],
                                                         [0, 1, 0]]))
        comp_of = {(tx, ty): int(lab[yi[ty], xi[tx]]) for tx, ty in accepted}
        comps = defaultdict(list)
        for (tx, ty) in accepted:
            comps[comp_of[(tx, ty)]].append((tx, ty))
        return list(comps.values())

    def cut_components(self, accepted: List[Pos], cuts, tw: int, th: int,
                       cut_thr: float = 0.5):
        """Multi-cut CCM: two grid-adjacent accepted tiles merge ONLY when a
        border-cut head fires on their shared border (right-cut of the left
        tile OR left-cut of the right tile; bottom/top vertically). This
        merges exactly the tiles a truncated target spans, instead of every
        grid-adjacent block. Returns components of (tx, ty)."""
        if not accepted:
            return []
        acc = set(accepted)
        parent = {t: t for t in accepted}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (tx, ty) in accepted:
            c = cuts.get((tx, ty), [0.0] * 4)
            nb = (tx + tw, ty)                      # right neighbour
            if nb in acc:
                nc = cuts.get(nb, [0.0] * 4)
                if c[1] >= cut_thr or nc[0] >= cut_thr:
                    union((tx, ty), nb)
            nb = (tx, ty + th)                      # bottom neighbour
            if nb in acc:
                nc = cuts.get(nb, [0.0] * 4)
                if c[3] >= cut_thr or nc[2] >= cut_thr:
                    union((tx, ty), nb)
        comps = defaultdict(list)
        for t in accepted:
            comps[find(t)].append(t)
        return list(comps.values())

    def merged_rect(self, comp: List[Pos], img_h: int, img_w: int,
                    tw: int | None = None, th: int | None = None):
        """Bounding rect (x1,y1,x2,y2) of a multi-tile component, clipped."""
        min_r = min(ty for _, ty in comp)
        max_r = max(ty for _, ty in comp)
        min_c = min(tx for tx, _ in comp)
        max_c = max(tx for tx, _ in comp)
        tw = tw or self.cfg.tile
        th = th or self.cfg.tile
        x1, y1 = min_c, min_r
        x2, y2 = min(max_c + tw, img_w), min(max_r + th, img_h)
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    def build_jobs(self, img, accepted: List[Pos], with_small_tiles=True,
                   timer=None):
        """Build detection jobs [(crop, offset_x, offset_y)] from components.

        with_small_tiles=True: multi-tile comps also emit individual tile
        crops (small-target recall, Scheme A+). False: merged crop only.
        """
        t0 = time.time()
        H, W = img.shape[:2]
        t = self.cfg.tile
        jobs = []
        for comp in self.components(accepted):
            if len(comp) == 1:
                tx, ty = comp[0]
                jobs.append((img[ty:min(ty + t, H), tx:min(tx + t, W)], tx, ty))
            else:
                x1, y1, x2, y2 = self.merged_rect(comp, H, W)
                crop = img[y1:y2, x1:x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    jobs.append((crop, x1, y1))            # large targets
                if with_small_tiles:
                    for tx, ty in comp:                     # small targets
                        jobs.append((img[ty:min(ty + t, H), tx:min(tx + t, W)],
                                     tx, ty))
        if timer is not None:
            timer['cc'] += time.time() - t0
        return jobs