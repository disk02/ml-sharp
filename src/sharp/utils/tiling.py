"""Utilities for tiled inference and camera compensation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tile:
    """Axis-aligned integer tile bounds in pixel coordinates.

    Coordinates follow (x0, y0, x1, y1) with a half-open interval:
    x in [x0, x1), y in [y0, y1).
    """

    x0: int
    y0: int
    x1: int
    y1: int


def make_tiles(W: int, H: int, tile_size: int, overlap: float) -> list[Tile]:
    """Generate a grid of overlapping tiles that cover an image.

    Args:
        W: Full image width in pixels.
        H: Full image height in pixels.
        tile_size: Nominal tile edge length in pixels (square tiles).
        overlap: Fractional overlap in [0.0, 0.5).
    """
    if W <= 0 or H <= 0:
        raise ValueError("W and H must be positive.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap < 0.0 or overlap >= 0.5:
        raise ValueError("overlap must be in [0.0, 0.5).")

    if tile_size >= W or tile_size >= H:
        return [Tile(0, 0, W, H)]

    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    def _starts(full: int) -> list[int]:
        starts: list[int] = []
        pos = 0
        while pos + tile_size < full:
            starts.append(pos)
            pos += stride
        final_start = max(0, full - tile_size)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        return starts

    x_starts = _starts(W)
    y_starts = _starts(H)

    tiles: list[Tile] = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(W, x0 + tile_size)
            y1 = min(H, y0 + tile_size)
            x0_tile = x0
            y0_tile = y0
            if x1 - x0_tile < tile_size:
                x0_tile = max(0, x1 - tile_size)
            if y1 - y0_tile < tile_size:
                y0_tile = max(0, y1 - tile_size)
            tiles.append(Tile(x0_tile, y0_tile, x1, y1))

    return tiles


def shift_intrinsics_for_tile(
    K_full: torch.Tensor, x0: float, y0: float
) -> torch.Tensor:
    """Shift intrinsics for a tile offset in full-image coordinates."""
    if K_full.shape != (3, 3):
        raise ValueError("K_full must have shape (3, 3).")
    K_tile = K_full.clone()
    K_tile[0, 2] = K_tile[0, 2] - x0
    K_tile[1, 2] = K_tile[1, 2] - y0
    return K_tile


def scale_intrinsics_for_resize(
    K: torch.Tensor, src_wh: tuple[int, int], dst_wh: tuple[int, int]
) -> torch.Tensor:
    """Scale intrinsics to match a resize from src_wh to dst_wh (width, height)."""
    if K.shape != (3, 3):
        raise ValueError("K must have shape (3, 3).")
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError("src_wh and dst_wh must be positive.")
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    K_scaled = K.clone()
    K_scaled[0, :] = K_scaled[0, :] * scale_x
    K_scaled[1, :] = K_scaled[1, :] * scale_y
    return K_scaled
