"""PO / APO - tiling strategies.

PO (Plain Overlap)  : fixed stride grid, stride = tile*(1-overlap). SAHI-like
                      vanilla overlap tiling. Returns all tile positions.
APO (Adaptive)      : non-overlapping base grid (stride=tile); for each
                      ACCEPTED tile, expand toward the side(s) where accepted
                      neighbors exist (or where content sits near the border),
                      so only the target side gets overlap. Isolated tiles are
                      sent as-is.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from .config import AGCMConfig

Pos = Tuple[int, int]


def plain_grid(W: int, H: int, tile: int, overlap: float) -> List[Pos]:
    """Vanilla fixed-stride tiling (PO). stride = tile*(1-overlap)."""
    stride = max(1, int(tile * (1.0 - overlap)))
    tiles = set()
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            tx, ty = min(x0, max(0, W - tile)), min(y0, max(0, H - tile))
            tiles.add((tx, ty))
    if H > tile and (H - tile) % stride != 0:
        for x0 in range(0, W, stride):
            tiles.add((min(x0, max(0, W - tile)), H - tile))
    if W > tile and (W - tile) % stride != 0:
        for y0 in range(0, H, stride):
            tiles.add((W - tile, min(y0, max(0, H - tile))))
    if W > tile and H > tile:
        tiles.add((W - tile, H - tile))
    return sorted(tiles)


def adaptive_tile(dim: int, base: int = 640,
                  lo: float = 0.8, hi: float = 1.2) -> int:
    """Approximate-divisor tile size: n = round(dim/base) tiles along the
    dimension, tile = ceil(dim/n) so the image is fully covered (the last
    tile is clipped by the crop). Keeps tile within [base*lo, base*hi] by
    adjusting n. Preferable to strict divisibility: yields near-square
    tiles (e.g. 5456 -> 607 with 9 cols) instead of 682x640 grids.
    The adjustment iterates (up to 8 rounds) so the returned tile never
    falls outside the clamp window (e.g. VisDrone 1360x765 used to give
    454x383 with the one-shot adjustment; 788 gave an out-of-range 788)."""
    lo_i, hi_i = int(base * lo), int(base * hi)
    n = max(1, int(round(dim / base)))
    t = int(math.ceil(dim / n))
    for _ in range(8):
        if t < lo_i:                     # tile too small -> fewer tiles
            n = max(1, dim // lo_i)
        elif t > hi_i:                   # tile too large -> more tiles
            n = max(1, dim // hi_i) + 1
        else:
            break
        t = int(math.ceil(dim / n))
    return max(lo_i, min(hi_i, t))       # hard clamp as last resort


def int_grid(W: int, H: int, tw: int, th: int) -> List[Pos]:
    """Non-overlapping integer grid: stride=(tw, th). adaptive_tile already
    guarantees n*t >= dim (t = ceil(dim/n)), so the last row/column of the
    plain stride grid covers the image edge (clipped by the crop). It must
    NOT add edge-aligned duplicates: those sit 1-7 px away from the regular
    grid line and (a) re-detect the same edge strip and (b) break CCM
    4-connectivity (neighbours are detected by an exact +/-tw/th offset),
    splitting fully-accepted scenes into many components."""
    tiles = []
    for y0 in range(0, H, th):
        for x0 in range(0, W, tw):
            tiles.append((x0, y0))
    return tiles


class PlainOverlap:
    """PO tiling: full fixed-overlap grid (no gating)."""

    def __init__(self, cfg: AGCMConfig):
        self.cfg = cfg

    def tiles(self, W: int, H: int) -> List[Pos]:
        return plain_grid(W, H, self.cfg.tile, self.cfg.overlap)


class AdaptiveOverlap:
    """APO tiling: base non-overlapping grid, expand accepted tiles toward
    accepted-neighbor side(s) only.

    Expansion semantics (matching the paper):
      - right  neighbor accepted -> expand 20% to the right
      - below  neighbor accepted -> expand 20% downward
      - left/up similarly
      - isolated accepted tile   -> no expansion (640x640 as-is)
    Returns jobs [(crop, ox, oy)] built directly (crops may be non-square;
    the detector crop-runner resizes to its input).
    """

    def __init__(self, cfg: AGCMConfig):
        self.cfg = cfg

    def tiles(self, W: int, H: int) -> List[Pos]:
        return plain_grid(W, H, self.cfg.tile, 0.0)   # base grid, no overlap

    def expanded_crops(self, img, accepted: List[Pos]):
        """Build APO crops for accepted tiles. Returns [(crop, ox, oy)]."""
        t = self.cfg.tile
        ex = int(round(self.cfg.tile * self.cfg.expand))   # e.g. 128 @640
        H, W = img.shape[:2]
        acc = set(accepted)
        jobs = []
        for tx, ty in accepted:
            has_r = (tx + t, ty) in acc
            has_d = (tx, ty + t) in acc
            has_l = (tx - t, ty) in acc
            has_u = (tx, ty - t) in acc
            if not (has_r or has_d or has_l or has_u):
                jobs.append((img[ty:min(ty + t, H), tx:min(tx + t, W)], tx, ty))
            else:
                x1 = max(0, tx - ex) if has_l else tx
                y1 = max(0, ty - ex) if has_u else ty
                x2 = min(tx + t + ex, W) if has_r else tx + t
                y2 = min(ty + t + ex, H) if has_d else ty + t
                jobs.append((img[y1:y2, x1:x2], x1, y1))
        return jobs