"""Depth loading utilities.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
import numpy as np

DEPTH_EXTENSION_PRIORITY = [".npy", ".png"]
SUPPORTED_DEPTH_EXTENSIONS = {".npy", ".png"}


def _ensure_hw(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 2:
        return depth
    if depth.ndim == 3 and depth.shape[0] == 1:
        return depth[0]
    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[..., 0]
    raise ValueError(f"Depth array must be HxW (or 1xHxW/HxWx1). Got {depth.shape}.")


def load_depth(path: Path, scale: float) -> np.ndarray:
    """Load a depth map as float32 HxW.

    PNG depth maps are expected to be uint16 and may store depth in arbitrary units.
    Use --depth-scale to convert into meters (e.g., 0.001 for millimeters).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".png":
        depth = iio.imread(path)
        if depth.dtype != np.uint16:
            raise ValueError(
                f"Expected uint16 PNG depth for {path}, got {depth.dtype} instead."
            )
    else:
        raise ValueError(
            f"Unsupported depth format {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_DEPTH_EXTENSIONS))}."
        )

    depth = _ensure_hw(depth).astype(np.float32) * scale
    non_finite = ~np.isfinite(depth)
    if non_finite.any():
        depth[non_finite] = 0.0
    return depth


def resolve_depth_for_image(image_path: Path, depth_path: Path) -> Path:
    """Resolve the depth file associated with an image."""
    if depth_path.is_file():
        return depth_path

    candidates = [
        depth_path / f"{image_path.stem}{ext}" for ext in DEPTH_EXTENSION_PRIORITY
    ]
    matches = [
        path
        for path in depth_path.iterdir()
        if path.is_file()
        and path.stem == image_path.stem
        and path.suffix.lower() in SUPPORTED_DEPTH_EXTENSIONS
    ]
    if not matches:
        candidates_str = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"No depth file found for image {image_path} in {depth_path}. "
            f"Expected one of: {candidates_str}."
        )

    matches_sorted = sorted(matches, key=lambda path: path.name)
    for ext in DEPTH_EXTENSION_PRIORITY:
        for match in matches_sorted:
            if match.suffix.lower() == ext:
                return match

    return matches_sorted[0]
