"""Contains `sharp render` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch

from sharp.rendering.gaussian_renderer import render_gaussians
from sharp.utils import camera
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import load_ply

LOGGER = logging.getLogger(__name__)


def _resolve_sbs_image_path(
    sbs_image: Path | None,
    scene_path: Path,
) -> Path | None:
    """Resolve per-scene SBS image output path.

    - If `sbs_image` is a directory-like path (no suffix), write <dir>/<scene_stem>_sbs.png.
    - If `sbs_image` is a file path, use it as-is.
    """
    if sbs_image is None:
        return None

    # Treat paths with no suffix as directories.
    if sbs_image.suffix == "":
        out_dir = sbs_image
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{scene_path.stem}_sbs").with_suffix(".png")

    sbs_image.parent.mkdir(parents=True, exist_ok=True)
    return sbs_image


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ply or a list of plys.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered videos.",
    required=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
@click.option(
    "--sbs-image",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path (file or directory) to save a single SBS frame image (PNG/JPG).",
)
@click.option(
    "--sbs-image-frame",
    type=int,
    default=0,
    help="Frame index to save for --sbs-image (default: 0).",
)
@click.option(
    "--align-stereo",
    is_flag=True,
    default=False,
    help="Enable stereo alignment and automatic overlap crop before saving SBS output.",
)
def render_cli(
    input_path: Path,
    output_path: Path,
    verbose: bool,
    sbs_image: Path | None,
    sbs_image_frame: int,
    align_stereo: bool,
):
    """Render 3DGS PLY files to SBS video (and optionally an SBS frame image)."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not torch.cuda.is_available():
        LOGGER.error("Rendering a checkpoint requires CUDA.")
        raise SystemExit(1)

    output_path.mkdir(exist_ok=True, parents=True)
    params = camera.TrajectoryParams()

    if input_path.suffix == ".ply":
        scene_paths = [input_path]
    elif input_path.is_dir():
        scene_paths = list(input_path.glob("*.ply"))
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        raise SystemExit(1)

    for scene_path in scene_paths:
        LOGGER.info("Rendering %s", scene_path)
        gaussians, metadata = load_ply(scene_path)

        render_gaussians(
            gaussians=gaussians,
            metadata=metadata,
            params=params,
            output_path=(output_path / scene_path.stem).with_suffix(".mp4"),
            sbs_image_path=_resolve_sbs_image_path(sbs_image, scene_path),
            sbs_image_frame=sbs_image_frame,
            align_stereo=align_stereo,
        )
