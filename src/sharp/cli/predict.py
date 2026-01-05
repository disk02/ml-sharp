"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils.depth_io import load_depth, resolve_depth_for_image
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from sharp.rendering.gaussian_renderer import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--sbs-image",
    type=click.Path(path_type=Path),
    default=None,
    flag_value="__AUTO__",
    show_default=False,
    help=(
        "Optional path to save a single SBS frame image (PNG/JPG). "
        "If provided without a value, saves to <output-path>/<image_stem>_sbs.png. "
        "If a directory path is provided, saves <dir>/<image_stem>_sbs.png."
    ),
)
@click.option(
    "--sbs-image-frame",
    type=int,
    default=0,
    show_default=True,
    help="Which frame index to save for --sbs-image.",
)
@click.option(
    "--save-ply/--no-save-ply",
    default=None,
    show_default=False,
    help="Whether to save the predicted Gaussians as .ply.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option(
    "--depth-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional path to a depth file or directory containing depth maps.",
)
@click.option(
    "--depth-format",
    type=click.Choice(["meters", "disparity"]),
    default="meters",
    show_default=True,
    help="Depth input format for --depth-path.",
)
@click.option(
    "--depth-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Scale factor applied to loaded depth values.",
)
@click.option(
    "--depth-fill",
    type=click.Choice(["override_only", "monodepth_fallback"]),
    default="override_only",
    show_default=True,
    help="How to fill invalid depth override pixels.",
)
@click.option(
    "--depth-calibration",
    type=click.Choice(["none", "per_image", "per_sequence"]),
    default="none",
    show_default=True,
    help="Optional calibration mode for normalized depth overrides.",
)
@click.option(
    "--depth-calibration-percentiles",
    type=(float, float),
    default=(10.0, 90.0),
    show_default=True,
    help="Percentile range for depth override calibration.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    sbs_image: Path | None,
    sbs_image_frame: int,
    save_ply: bool | None,
    device: str,
    depth_path: Path | None,
    depth_format: str,
    depth_scale: float,
    depth_fill: str,
    depth_calibration: str,
    depth_calibration_percentiles: tuple[float, float],
    verbose: bool,
):
    """Predict Gaussians from input images."""
    # Depth override examples:
    # images/
    #   img_0001.jpg
    #   img_0002.jpg
    # depth/
    #   img_0001.png   (uint16, mm; use --depth-scale 0.001)
    #   img_0002.png
    #
    # sharp predict --input-path images --depth-path depth --depth-format meters --depth-scale 0.001
    # sharp predict --input-path images --depth-path depth --depth-format disparity
    # sharp predict --input-path images --depth-path depth --depth-calibration per_image \
    #   --depth-calibration-percentiles 10 90
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))
    if depth_path is not None:
        if depth_path.is_file() and len(image_paths) > 1:
            raise click.UsageError(
                f"--depth-path {depth_path} is a file but multiple images were found."
            )
        LOGGER.info("Depth override enabled (%s).", depth_format)

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    want_sbs_image = sbs_image is not None
    want_video = with_rendering
    if save_ply is None and want_sbs_image:
        effective_save_ply = False
    elif save_ply is None and not want_sbs_image:
        effective_save_ply = True
    else:
        effective_save_ply = bool(save_ply)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        depth_override = None
        if depth_path is not None:
            depth_file = (
                depth_path
                if depth_path.is_file()
                else resolve_depth_for_image(image_path, depth_path)
            )
            depth_override = load_depth(depth_file, depth_scale)
            LOGGER.info("Loaded depth override from %s", depth_file)

        gaussians = predict_image(
            gaussian_predictor,
            image,
            f_px,
            torch.device(device),
            depth_override=depth_override,
            depth_override_is_disparity=(depth_format == "disparity"),
            depth_override_fill_mode=depth_fill,
            depth_override_calibration=depth_calibration,
            depth_override_calibration_percentiles=depth_calibration_percentiles,
        )

        if effective_save_ply:
            LOGGER.info("Saving 3DGS to %s", output_path)
            save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")
        else:
            if save_ply is None and want_sbs_image:
                LOGGER.info(
                    "Skipping .ply save because --sbs-image was requested (use --save-ply to override)."
                )
            else:
                LOGGER.info("Skipping .ply save because --no-save-ply was requested.")

        # Determine SBS image output path (optional)
        sbs_image_path: Path | None = None
        if want_sbs_image:
            sbs_out = Path(sbs_image)
            # Support `--sbs-image` with no value: default to output directory
            if sbs_out.name == "__AUTO__":
                sbs_image_path = output_path / f"{image_path.stem}_sbs.png"
            else:
                # If a directory path is provided (no suffix), place an image per input
                if sbs_out.suffix == "":
                    sbs_image_path = sbs_out / f"{image_path.stem}_sbs.png"
                else:
                    # If multiple inputs are processed and a single file path is given, avoid overwrites
                    if len(image_paths) > 1:
                        sbs_image_path = sbs_out.with_name(
                            f"{sbs_out.stem}_{image_path.stem}{sbs_out.suffix}"
                        )
                    else:
                        sbs_image_path = sbs_out

        if want_video or sbs_image_path is not None:
            if want_video:
                output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
                LOGGER.info("Rendering trajectory to %s", output_video_path)
            else:
                # Placeholder path; render_gaussians will not write video when sbs_image_path is set.
                output_video_path = (output_path / image_path.stem).with_suffix(".mp4")

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")

            render_gaussians(
                gaussians=gaussians,
                metadata=metadata,
                output_path=output_video_path,
                sbs_image_path=sbs_image_path,
                sbs_image_frame=sbs_image_frame,
            )


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    depth_override: np.ndarray | None = None,
    depth_override_is_disparity: bool = False,
    depth_override_fill_mode: str = "override_only",
    depth_override_calibration: str = "none",
    depth_override_calibration_percentiles: tuple[float, float] = (10.0, 90.0),
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    depth_override_pt = None
    if depth_override is not None:
        depth_override_pt = torch.from_numpy(depth_override).to(device)
        if depth_override_pt.dim() == 2:
            depth_override_pt = depth_override_pt.unsqueeze(0)

    gaussians_ndc = predictor(
        image_resized_pt,
        disparity_factor,
        depth_override=depth_override_pt,
        depth_override_is_disparity=depth_override_is_disparity,
        depth_override_fill_mode=depth_override_fill_mode,
        depth_override_calibration=depth_override_calibration,
        depth_override_calibration_percentiles=depth_override_calibration_percentiles,
    )

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians
