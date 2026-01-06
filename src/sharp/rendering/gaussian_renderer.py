"""Rendering utilities for Gaussian splats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp.utils import camera, gsplat, io
from sharp.utils.gaussians import Gaussians3D, SceneMetaData


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    sbs_image_path: Path | None = None,
    sbs_image_frame: int = 0,
    align_stereo: bool = False,
) -> None:
    """Render a single gaussian checkpoint file.

    Stereo alignment is disabled by default and only occurs when align_stereo=True
    is explicitly passed by the caller.
    """
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
            # Convert torch -> numpy (H,W,3) uint8
            color_l_np = color_l.detach().cpu().numpy()
            color_r_np = color_r.detach().cpu().numpy()

            # NOTE: Alignment must be explicitly enabled by the caller.
            if align_stereo:
                # Import lazily so OpenCV isn't required unless alignment is used.
                try:
                    from sharp.utils.stereo_align import AlignParams, auto_align_and_crop
                except ImportError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Stereo auto-alignment requires OpenCV. Install opencv-python "
                        "(or opencv-python-headless)."
                    ) from e

                # Auto-align + auto-crop the stereo pair, then re-pack SBS for output.
                align_params = AlignParams()
                color_l_aligned, color_r_aligned, _meta = auto_align_and_crop(
                    color_l_np, color_r_np, params=align_params
                )
            else:
                color_l_aligned, color_r_aligned = color_l_np, color_r_np

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
