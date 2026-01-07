"""Rendering utilities for Gaussian splats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp.utils import camera, gsplat, io
from sharp.utils.gaussians import Gaussians3D, SceneMetaData
from sharp.utils.metrics import Metrics


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    sbs_image_path: Path | None = None,
    sbs_image_frame: int = 0,
    align_crop: bool = False,
    metrics: Metrics | None = None,
) -> None:
    """Render a single gaussian checkpoint file."""
    if metrics:
        metrics.inc("render_calls")
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
        if metrics:
            metrics.inc("render_frames")
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

            if align_crop:
                # Import lazily so OpenCV isn't required unless --align-crop is used.
                try:
                    from sharp.utils.stereo_align import AlignParams, auto_align_and_crop
                except ImportError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Stereo auto-alignment requires OpenCV. Install opencv-python "
                        "(or opencv-python-headless), or omit --align-crop."
                    ) from e

                # Auto-align + auto-crop the stereo pair, then re-pack SBS for output.
                align_params = AlignParams()
                color_l_np, color_r_np, _meta = auto_align_and_crop(
                    color_l_np, color_r_np, params=align_params
                )

            color_sbs_np = np.concatenate((color_l_np, color_r_np), axis=1)
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


def render_gaussians_pred_space(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    unprojection_matrix: torch.Tensor,
    params: camera.TrajectoryParams | None = None,
    sbs_image_path: Path | None = None,
    sbs_image_frame: int = 0,
    align_crop: bool = False,
    metrics: Metrics | None = None,
) -> None:
    """Render predicted-space Gaussians by folding unprojection into the camera."""
    if metrics:
        metrics.inc("render_calls")
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
    u_pred_to_world = unprojection_matrix.to(device=device, dtype=torch.float32)

    mean_vectors = gaussians.mean_vectors
    if mean_vectors.ndim == 3:
        mean_vectors = mean_vectors[0]
    num_points = int(mean_vectors.shape[0])
    sample_size = min(16384, num_points)
    sample_idx = torch.randint(0, num_points, (sample_size,), device=mean_vectors.device)
    mean_sample = mean_vectors[sample_idx]
    mean_sample_world = mean_sample @ u_pred_to_world[:3, :3].T + u_pred_to_world[:3, 3]
    mean_sample_world = mean_sample_world.unsqueeze(0)
    quaternions = gaussians.quaternions
    singular_values = gaussians.singular_values
    colors = gaussians.colors
    opacities = gaussians.opacities
    if quaternions.ndim == 3:
        quaternions = quaternions[0][sample_idx].unsqueeze(0)
        singular_values = singular_values[0][sample_idx].unsqueeze(0)
        colors = colors[0][sample_idx].unsqueeze(0)
        opacities = opacities[0][sample_idx].unsqueeze(0)
    gaussians_camera_sample = Gaussians3D(
        mean_vectors=mean_sample_world,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )
    camera_model = camera.create_camera_model(
        gaussians_camera_sample, intrinsics, resolution_px=metadata.resolution_px
    )

    baseline = 0.065
    params.num_repeats = 3
    if sbs_image_path is not None:
        params.type = "static"

    trajectory = camera.create_eye_trajectory(
        gaussians_camera_sample, params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)

    video_writer = io.VideoWriter(output_path) if sbs_image_path is None else None

    for frame_idx, eye_mid in enumerate(trajectory):
        if metrics:
            metrics.inc("render_frames")
        eye_position_l = eye_mid.clone()
        eye_position_l[0] -= baseline * 0.5
        eye_position_r = eye_mid.clone()
        eye_position_r[0] += baseline * 0.5

        camera_info = camera_model.compute(eye_position_l)
        extrinsics_l = camera_info.extrinsics.to(device) @ u_pred_to_world
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=extrinsics_l[None],
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color_l = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(
            dtype=torch.uint8
        )
        depth_l = rendering_output.depth[0]

        camera_info = camera_model.compute(eye_position_r)
        extrinsics_r = camera_info.extrinsics.to(device) @ u_pred_to_world
        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=extrinsics_r[None],
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color_r = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(
            dtype=torch.uint8
        )
        depth_r = rendering_output.depth[0]

        color = torch.cat((color_l, color_r), dim=1)

        if sbs_image_path is not None and frame_idx == sbs_image_frame:
            color_l_np = color_l.detach().cpu().numpy()
            color_r_np = color_r.detach().cpu().numpy()

            if align_crop:
                try:
                    from sharp.utils.stereo_align import AlignParams, auto_align_and_crop
                except ImportError as e:  # pragma: no cover
                    raise RuntimeError(
                        "Stereo auto-alignment requires OpenCV. Install opencv-python "
                        "(or opencv-python-headless), or omit --align-crop."
                    ) from e

                align_params = AlignParams()
                color_l_np, color_r_np, _meta = auto_align_and_crop(
                    color_l_np, color_r_np, params=align_params
                )

            color_sbs_np = np.concatenate((color_l_np, color_r_np), axis=1)
            img = Image.fromarray(color_sbs_np, mode="RGB")
            sbs_image_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(sbs_image_path)
            break

        if video_writer is not None:
            depth = torch.cat((depth_l, depth_r), dim=0)
            video_writer.add_frame(color, depth)

    if video_writer is not None:
        video_writer.close()
