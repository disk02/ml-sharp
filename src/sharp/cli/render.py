"""Contains `sharp render` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from sharp.utils import camera, gsplat, io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply

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
def render_cli(
    input_path: Path,
    output_path: Path,
    verbose: bool,
    sbs_image: Path | None,
    sbs_image_frame: int,
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
        )


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    sbs_image_path: Path | None = None,
    sbs_image_frame: int = 0,
) -> None:
    """Render a single gaussian checkpoint file."""
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px

    if params is None:
        params = camera.TrajectoryParams()

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering a checkpoint requires CUDA.")

    device = torch.device("cuda")

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
    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px
    )

    # Stereo baseline (world units in the model's coordinate system).
    baseline = 0.065

    # Number of camera animation loops.
    params.num_repeats = 3

    # Use static trajectory for SBS image mode
    if sbs_image_path is not None:
        params.type = "static"

    trajectory = camera.create_eye_trajectory(
        gaussians, params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)

    # If only rendering SBS image, don't create video writer
    video_writer = io.VideoWriter(output_path) if sbs_image_path is None else None

    for frame_idx, eye_mid in enumerate(trajectory):
        # Treat the trajectory as the *midpoint* between eyes so the midpoint stays centered.
        eye_position_l = eye_mid.clone()
        eye_position_l[0] -= baseline * 0.5
        eye_position_r = eye_mid.clone()
        eye_position_r[0] += baseline * 0.5

        # Left view
        camera_info = camera_model.compute(eye_position_l)
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color_l = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(
            dtype=torch.uint8
        )
        depth_l = rendering_output.depth[0]

        # Right view
        camera_info = camera_model.compute(eye_position_r)
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color_r = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(
            dtype=torch.uint8
        )
        depth_r = rendering_output.depth[0]

        # Pack the left and right views into SBS format.
        color = torch.cat((color_l, color_r), dim=1)

        # Write SBS frame image if requested
        if sbs_image_path is not None and frame_idx == sbs_image_frame:
            # Import lazily so OpenCV isn't required unless --sbs-image is used.
            try:
                from sharp.utils.stereo_align import AlignParams, auto_align_and_crop
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "Stereo auto-alignment requires OpenCV. Install opencv-python "
                    "(or opencv-python-headless)."
                ) from e

            # Convert torch -> numpy (H,W,3) uint8
            color_l_np = color_l.detach().cpu().numpy()
            color_r_np = color_r.detach().cpu().numpy()

            # Auto-align + auto-crop the stereo pair, then re-pack SBS for output.
            align_params = AlignParams()
            color_l_aligned, color_r_aligned, _meta = auto_align_and_crop(
                color_l_np, color_r_np, params=align_params
            )

            color_sbs_np = np.concatenate((color_l_aligned, color_r_aligned), axis=1)
            img = Image.fromarray(color_sbs_np, mode="RGB")
            sbs_image_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(sbs_image_path)
            break

        # Only add to video if video writer was created
        if video_writer is not None:
            depth = torch.cat((depth_l, depth_r), dim=0)
            video_writer.add_frame(color, depth)

    # Only close video writer if it was created
    if video_writer is not None:
        video_writer.close()
