"""GF - Adaptive Gated Filtering.

Wraps the lightweight gate CNN (CustomTiny architecture) and provides:
  - build_net(): construct the network (SameNet hard-coded; training code lives
    in scripts/, inference only here)
  - score_image(): batch score all 224x224 gate patches of an image
  - filter_tiles(): score tiles and return accepted ones above `thr`

Two scoring backends, identical decisions (verified in experiments):
  - score_tiles_naive : per-tile resize + one batched forward (v1)
  - score_tiles_fast  : whole-image downscale once, slice patches (v2/v3)
"""
from __future__ import annotations

import time
import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import AGCMConfig


def build_net(n_out: int = 1) -> nn.Module:
    """CustomTiny gate: 5-conv mini CNN + global pool -> n_out logits.
    1 output = has-target (original gate); 5 outputs = multi-cut gate
    [has, cut_l, cut_r, cut_t, cut_b] (table 3)."""
    m = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(256, n_out))
    return m


class _SEBlock(nn.Module):
    def __init__(self, c, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c // reduction), nn.SiLU(),
            nn.Linear(c // reduction, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = _SEBlock(out_c, reduction=4)
        self.pool = nn.MaxPool2d(2)
        self.skip = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c \
            else nn.Identity()
        self.bn_skip = nn.BatchNorm2d(out_c) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.bn_skip(self.skip(x))
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.silu(out + identity)
        return self.pool(out)


def build_tinyresnet_net(n_out: int = 1) -> nn.Module:
    """TinyResNet gate: 5 residual blocks (16..256) + SE attention; the
    has head and cut heads are SEPARATE linear layers over the shared
    256-d feature (multi-cut variant). ~1.3M params (vs 0.4M CustomTiny)."""
    channels = [16, 32, 64, 128, 256]
    layers, in_c = [], 3
    for c in channels:
        layers.append(_ResBlock(in_c, c))
        in_c = c
    net = nn.Module()
    net.features = nn.Sequential(*layers)
    net.pool = nn.AdaptiveAvgPool2d(1)
    net.flatten = nn.Flatten()
    net.drop = nn.Dropout(0.15)
    if n_out == 5:
        net.has_head = nn.Linear(256, 1)
        net.cut_head = nn.Linear(256, 4)
    else:
        net.head = nn.Linear(256, 1)

    def forward(x):
        f = net.drop(net.flatten(net.pool(net.features(x))))
        if hasattr(net, 'cut_head'):
            return torch.cat([net.has_head(f), net.cut_head(f)], dim=1)
        return net.head(f)

    net.forward = forward
    return net


class GateFilter:
    """GF module: gate net + batched tile scoring + threshold decision.

    Supports both the original 1-output gate (has-target) and the multi-cut
    gate (has + 4 border-cut heads). Output width is auto-detected from the
    checkpoint. When multi_cut, filter_tiles() also fills self.cuts with
    per-tile cut probabilities {(tx,ty): [cut_l, cut_r, cut_t, cut_b]} for
    cut-driven APO expansion."""

    def __init__(self, weight: str, cfg: AGCMConfig, device: str = 'cuda:0'):
        self.cfg = cfg
        self.dev = device
        sd = torch.load(weight, map_location='cpu')
        # output width: TinyResNet multi-cut = has_head + cut_head rows;
        # otherwise the last 2D weight's rows (CustomTiny sequential head)
        if 'cut_head.weight' in sd:
            n_out = (sd['has_head.weight'].shape[0]
                     + sd['cut_head.weight'].shape[0])
        else:
            n_out = next(v.shape[0] for k, v in reversed(list(sd.items()))
                         if v.dim() == 2 and k.endswith('.weight'))
        if any(k.startswith('features.') for k in sd):
            self.net = build_tinyresnet_net(n_out)
        else:
            self.net = build_net(n_out)
        self.net.load_state_dict(sd)
        self.net.eval().to(device)
        self.n_out = n_out
        self.multi_cut = n_out == 5
        self.cuts = None                   # filled by filter_tiles()
        self.mean = torch.tensor(cfg.gate_mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(cfg.gate_std).view(1, 3, 1, 1).to(device)

    # ------------------------------------------------------------------
    def _patch(self, img: np.ndarray, tx: int, ty: int,
               tw: int | None = None, th: int | None = None) -> np.ndarray:
        """gate_size x gate_size patch over the ACTUAL tile region
        (tw x th; adaptive grids may differ from cfg.tile), BGR,
        unnormalized."""
        tw = tw or self.cfg.tile
        th = th or self.cfg.tile
        c = img[ty:ty + th, tx:tx + tw]
        c = cv2.resize(c, (self.cfg.gate_size, self.cfg.gate_size),
                       interpolation=cv2.INTER_AREA)
        return c

    def _score_naive(self, img, tiles, timer=None, tile_hw=None):
        """v1: per-tile resize, one batched forward. Returns (N, C) sigmoid.
        tile_hw=(tw,th): extract patches over the actual adaptive tile."""
        if not tiles:
            return np.zeros((0, self.n_out))
        tw, th = tile_hw or (None, None)
        t0 = time.time()
        crops = [self._patch(img, tx, ty, tw, th) for tx, ty in tiles]
        x = torch.from_numpy(np.stack(crops)[..., ::-1].copy()) \
            .permute(0, 3, 1, 2).float().div_(255.).to(self.dev)
        x = (x - self.mean) / self.std
        if timer is not None:
            timer['gate_pre'] += time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            logits = self.net(x.to(self.dev))
        if timer is not None:
            timer['gate_inf'] += time.time() - t0
        return torch.sigmoid(logits).float().cpu().numpy()

    def score_tiles_naive(self, img, tiles, timer=None, tile_hw=None):
        """v1: per-tile resize, one batched forward (has score, 1-D)."""
        return self._score_naive(img, tiles, timer, tile_hw)[:, 0].tolist()

    def _score_fast(self, img, tiles, timer=None, tile_hw=None):
        """v3 body: downscale whole image once (gate_size / actual tile),
        slice patches on GPU. Returns (N, C) sigmoid.

        tile_hw=(tw,th) is the ACTUAL adaptive grid step; downscale and
        window offsets follow it so each patch covers exactly its tile
        (a fixed cfg.tile window silently missed the bottom/right strips
        whenever the adaptive tile exceeded 640 px).

        Downscale runs with F.interpolate on the GPU (torch bilinear) instead
        of cv2.resize on the CPU: verified decision-identical (interp vs
        cv2.INTER_LINEAR, 720/720 tiles agree) but ~3x faster and immune to
        CPU contention from concurrent training. The image copy H,W,3 uint8
        -> GPU is a single ~2ms transfer; slicing is zero-copy views.
        """
        if not tiles:
            return np.zeros((0, self.n_out))
        import torch.nn.functional as F
        tw, th = tile_hw or (self.cfg.tile, self.cfg.tile)
        H, W = img.shape[:2]
        scale_x = self.cfg.gate_size / tw
        scale_y = self.cfg.gate_size / th
        t0 = time.time()
        sh = max(1, int(round(H * scale_y)))
        sw = max(1, int(round(W * scale_x)))
        if sh < self.cfg.gate_size or sw < self.cfg.gate_size:
            return self._score_naive(img, tiles, timer, tile_hw)  # fallback
        x = torch.from_numpy(np.ascontiguousarray(img)).to(self.dev)  # H,W,3
        x = x.permute(2, 0, 1)[None].float().div_(255.)               # 1,3,H,W
        small = F.interpolate(x, size=(sh, sw), mode='bilinear',
                              align_corners=False)
        small = small.flip(1)                                    # BGR -> RGB
        small = (small - self.mean) / self.std
        windows = []
        for tx, ty in tiles:
            sx = min(int(round(tx * scale_x)), sw - self.cfg.gate_size)
            sy = min(int(round(ty * scale_y)), sh - self.cfg.gate_size)
            windows.append(small[:, :, sy:sy + self.cfg.gate_size,
                             sx:sx + self.cfg.gate_size])
        xt = torch.cat(windows, dim=0)
        if timer is not None:
            timer['gate_pre'] += time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            logits = self.net(xt)
        if timer is not None:
            timer['gate_inf'] += time.time() - t0
        return torch.sigmoid(logits).float().cpu().numpy()

    def score_tiles_fast(self, img, tiles, timer=None, tile_hw=None):
        """v3: whole-image downscale, GPU patches (has score, 1-D)."""
        return self._score_fast(img, tiles, timer, tile_hw)[:, 0].tolist()

    def score_tiles_full(self, img, tiles, fast=True, timer=None,
                         tile_hw=None):
        """Full (N, C) sigmoid rows: [has, cut_l, cut_r, cut_t, cut_b]
        for multi-cut gates (C=1 for the original has-only gate)."""
        fn = self._score_fast if fast else self._score_naive
        return fn(img, tiles, timer, tile_hw)

    # ------------------------------------------------------------------
    def filter_tiles(self, img, tiles, fast=True, thr=None, timer=None,
                     tile_hw=None):
        """Score tiles and return (accepted[(tx,ty)...], has_scores, rate).

        tile_hw=(tw,th): actual adaptive grid step, so each gate patch covers
        exactly its tile region (a fixed cfg.tile window mismatched adaptive
        grids: strips beyond 640 px were never gated).

        For multi-cut gates also fills self.cuts = {(tx,ty): [cut_l, cut_r,
        cut_t, cut_b]} (sigmoid probs) so the caller can expand APO on the
        exact borders that truncate a target.
        """
        thr = self.cfg.thr if thr is None else thr
        full = self.score_tiles_full(img, tiles, fast, timer, tile_hw)
        has_scores = full[:, 0].tolist()
        accepted = [(tx, ty) for (tx, ty), s in zip(tiles, has_scores)
                    if s >= thr]
        if self.multi_cut:
            self.cuts = {t: [float(v) for v in row[1:]]
                         for t, row in zip(tiles, full)}
        else:
            self.cuts = None
        accept_rate = len(accepted) / max(len(tiles), 1)
        return accepted, has_scores, accept_rate