"""Rendering utilities for Gaussian splats."""

from __future__ import annotations

import dataclasses
import io as py_io
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch
from PIL import Image

from sharp.utils import camera, gsplat, io
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    build_camera_stats_scene,
    prune_gaussians,
)
from sharp.utils.metrics import Metrics, RenderTiming


def _pinhole_intrinsics(
    f_px: float,
    width_px: int,
    height_px: int,
    device: torch.device,
) -> torch.Tensor:
    """Build pinhole intrinsics from focal length and resolution."""
    return torch.tensor(
        [
            [f_px, 0, (width_px - 1) / 2.0, 0],
            [0, f_px, (height_px - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )


def _timed_cpu(render_timing: RenderTiming | None, stage: str) -> ContextManager[None]:
    """CPU stage timer; no-op when metrics are disabled."""
    if render_timing is None:
        return nullcontext()
    return render_timing.timed_cpu(stage)


def _timed_gpu(render_timing: RenderTiming | None, stage: str) -> ContextManager[None]:
    """GPU stage timer; no-op when metrics are disabled."""
    if render_timing is None:
        return nullcontext()
    return render_timing.gpu_event_timer(stage)


def _save_sbs_image(
    img: Image.Image | np.ndarray,
    sbs_image_path: Path,
    sbs_image_format: str | None,
    sbs_jpeg_quality: int,
    render_timing: RenderTiming | None,
    sbs_async_writer: Any | None = None,
) -> None:
    if sbs_async_writer is not None:
        with _timed_cpu(render_timing, "render_encode_write"):
            sbs_async_writer.submit(
                sbs_image_path,
                img,
                sbs_image_format=sbs_image_format,
                sbs_jpeg_quality=sbs_jpeg_quality,
            )
        return

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img, mode="RGB")
    img_format = Image.registered_extensions().get(sbs_image_path.suffix.lower())
    save_kwargs: dict[str, object] = {}
    if sbs_image_format == "jpg":
        img = img.convert("RGB")
        save_kwargs = {"quality": sbs_jpeg_quality, "subsampling": 0, "optimize": False}

    if img_format is None:
        with _timed_cpu(render_timing, "render_encode_write"):
            img.save(sbs_image_path, **save_kwargs)
        return

    with _timed_cpu(render_timing, "render_encode_compress"):
        bytes_io = py_io.BytesIO()
        img.save(bytes_io, format=img_format, **save_kwargs)
    with _timed_cpu(render_timing, "render_encode_write"):
        with sbs_image_path.open("wb") as file_handle:
            file_handle.write(bytes_io.getvalue())


def _finalize_sbs_frame(
    color_l: torch.Tensor,
    color_r: torch.Tensor,
    align_crop: bool,
    render_timing: RenderTiming | None,
) -> np.ndarray:
    """Pack rendered left/right frames into an SBS array, optionally aligning first."""
    with _timed_cpu(render_timing, "render_d2h_transfer"):
        height, width, _channels = color_l.shape
        color_sbs_u8 = torch.empty(
            (height, width * 2, 3), dtype=torch.uint8, device=color_l.device
        )
        color_sbs_u8[:, :width, :] = color_l
        color_sbs_u8[:, width:, :] = color_r
        color_sbs_np = color_sbs_u8.cpu().numpy()

    with _timed_cpu(render_timing, "render_encode_prepare"):
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
                color_sbs_np[:, :width, :],
                color_sbs_np[:, width:, :],
                params=align_params,
            )
            height, width, _channels = color_l_np.shape
            color_sbs_np = np.empty((height, width * 2, 3), dtype=np.uint8)
            color_sbs_np[:, :width, :] = color_l_np
            color_sbs_np[:, width:, :] = color_r_np
    return color_sbs_np


def _render_stereo_outputs(
    gaussians_device: Gaussians3D,
    camera_model: camera.PinholeCameraModel,
    trajectory: list[torch.Tensor],
    extrinsics_post: torch.Tensor | None,
    renderer: gsplat.GSplatRenderer,
    device: torch.device,
    output_path: Path,
    sbs_image_path: Path | None,
    sbs_image_format: str | None,
    sbs_jpeg_quality: int,
    sbs_image_frame: int,
    align_crop: bool,
    stereo_baseline: float,
    stereo_mode: camera.StereoMode,
    stereo_convergence_depth: float | None,
    stereo_convergence_norm: float | None,
    metrics: Metrics | None,
    sbs_async_writer: Any | None,
) -> None:
    """Render stereo frames from a trajectory, writing an SBS image and/or video.

    `trajectory` yields eye-midpoint camera poses. `extrinsics_post` optionally
    right-multiplies each eye's world-space extrinsics (predicted-space rendering
    folds the unprojection matrix in this way); pass `None` for world-space
    rendering. The stereo baseline is in world units so SBS outputs can match
    different capture rigs while keeping the midpoint fixed.
    """
    render_timing = metrics.render_timing if metrics else None

    # If only rendering SBS image, don't create video writer
    video_writer = io.VideoWriter(output_path) if sbs_image_path is None else None
    # SBS images don't use depth, so skip depth rendering when no video output is requested.
    want_depth = video_writer is not None

    for frame_idx, eye_mid in enumerate(trajectory):
        if metrics:
            metrics.inc("render_frames")
        # Stage mapping: setup = camera pose prep, pack_inputs = assemble tensors,
        # h2d_transfer = device copies, gpu_* = renderer/GPU post, d2h/output = readback + encode.
        if render_timing:
            render_timing.start_frame()
        with _timed_cpu(render_timing, "render_setup"):
            # Treat the trajectory as the *midpoint* between eyes so the midpoint stays centered.
            # Shared stereo geometry across world-space and predicted-space rendering.
            camera_info_l, camera_info_r = camera_model.compute_stereo_pair(
                eye_mid,
                baseline=stereo_baseline,
                stereo_mode=stereo_mode,
                stereo_convergence_depth=stereo_convergence_depth,
                stereo_convergence_norm=stereo_convergence_norm,
            )

        with _timed_cpu(render_timing, "render_pack_inputs"):
            intrinsics_l = camera_info_l.intrinsics[None]
            intrinsics_r = camera_info_r.intrinsics[None]

        with _timed_cpu(render_timing, "render_h2d_transfer"):
            if extrinsics_post is not None:
                # Apply the space transform before batching to keep per-path math identical.
                extrinsics_l = (camera_info_l.extrinsics.to(device) @ extrinsics_post)[None]
                extrinsics_r = (camera_info_r.extrinsics.to(device) @ extrinsics_post)[None]
            else:
                extrinsics_l = camera_info_l.extrinsics[None].to(device)
                extrinsics_r = camera_info_r.extrinsics[None].to(device)
            intrinsics_l = intrinsics_l.to(device)
            intrinsics_r = intrinsics_r.to(device)

        # Left view
        rendering_output_l = renderer(
            gaussians_device,
            extrinsics=extrinsics_l,
            intrinsics=intrinsics_l,
            image_width=camera_info_l.width,
            image_height=camera_info_l.height,
            want_depth=want_depth,
            render_timing=render_timing,
        )

        # Right view
        rendering_output_r = renderer(
            gaussians_device,
            extrinsics=extrinsics_r,
            intrinsics=intrinsics_r,
            image_width=camera_info_r.width,
            image_height=camera_info_r.height,
            want_depth=want_depth,
            render_timing=render_timing,
        )

        with _timed_gpu(render_timing, "render_gpu_raster_blend"):
            color_l = (
                rendering_output_l.color[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0
            ).to(dtype=torch.uint8)
            color_r = (
                rendering_output_r.color[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0
            ).to(dtype=torch.uint8)
            depth_l = rendering_output_l.depth[0] if want_depth else None
            depth_r = rendering_output_r.depth[0] if want_depth else None
            color = (
                torch.cat((color_l, color_r), dim=1) if video_writer is not None else None
            )

        # Write SBS frame image if requested
        if sbs_image_path is not None and frame_idx == sbs_image_frame:
            color_sbs_np = _finalize_sbs_frame(color_l, color_r, align_crop, render_timing)
            img = (
                color_sbs_np
                if sbs_async_writer is not None
                else Image.fromarray(color_sbs_np, mode="RGB")
            )
            _save_sbs_image(
                img,
                sbs_image_path,
                sbs_image_format,
                sbs_jpeg_quality,
                render_timing,
                sbs_async_writer,
            )
            if render_timing:
                render_timing.finalize_frame()
            break

        # Only add to video if video writer was created
        if video_writer is not None:
            depth = torch.cat((depth_l, depth_r), dim=0)
            video_writer.add_frame(color, depth, render_timing=render_timing)
        if render_timing:
            render_timing.finalize_frame()

    # Only close video writer if it was created
    if video_writer is not None:
        video_writer.close()


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    sbs_image_path: Path | None = None,
    sbs_image_format: str | None = None,
    sbs_jpeg_quality: int = 90,
    sbs_image_frame: int = 0,
    align_crop: bool = False,
    metrics: Metrics | None = None,
    sbs_async_writer: Any | None = None,
    stereo_baseline: float = camera.DEFAULT_STEREO_BASELINE,
    stereo_mode: camera.StereoMode = "toe_in",
    stereo_convergence_depth: float | None = None,
    stereo_convergence_norm: float | None = None,
    min_opacity: float = 0.0,
    min_scale: float = 0.0,
    max_splats: int | None = None,
    prune_score: str = "opacity_scale",
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
    gaussians = prune_gaussians(
        gaussians,
        min_opacity=min_opacity,
        min_scale=min_scale,
        max_splats=max_splats,
        score=prune_score,
    )
    gaussians_device = gaussians.to(device)

    intrinsics = _pinhole_intrinsics(f_px, width, height, device)
    camera_model = camera.create_camera_model(
        gaussians_device, intrinsics, resolution_px=metadata.resolution_px
    )

    # Number of camera animation loops.
    render_params = dataclasses.replace(params, num_repeats=3)

    # Use static trajectory for SBS image mode
    if sbs_image_path is not None:
        render_params = dataclasses.replace(render_params, type="static")
        sbs_image_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = camera.create_eye_trajectory(
        gaussians_device, render_params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)

    _render_stereo_outputs(
        gaussians_device=gaussians_device,
        camera_model=camera_model,
        trajectory=trajectory,
        extrinsics_post=None,
        renderer=renderer,
        device=device,
        output_path=output_path,
        sbs_image_path=sbs_image_path,
        sbs_image_format=sbs_image_format,
        sbs_jpeg_quality=sbs_jpeg_quality,
        sbs_image_frame=sbs_image_frame,
        align_crop=align_crop,
        stereo_baseline=stereo_baseline,
        stereo_mode=stereo_mode,
        stereo_convergence_depth=stereo_convergence_depth,
        stereo_convergence_norm=stereo_convergence_norm,
        metrics=metrics,
        sbs_async_writer=sbs_async_writer,
    )


def render_gaussians_pred_space(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    unprojection_matrix: torch.Tensor,
    params: camera.TrajectoryParams | None = None,
    sbs_image_path: Path | None = None,
    sbs_image_format: str | None = None,
    sbs_jpeg_quality: int = 90,
    sbs_image_frame: int = 0,
    align_crop: bool = False,
    metrics: Metrics | None = None,
    sbs_async_writer: Any | None = None,
    stereo_baseline: float = camera.DEFAULT_STEREO_BASELINE,
    stereo_mode: camera.StereoMode = "toe_in",
    stereo_convergence_depth: float | None = None,
    stereo_convergence_norm: float | None = None,
    min_opacity: float = 0.0,
    min_scale: float = 0.0,
    max_splats: int | None = None,
    prune_score: str = "opacity_scale",
) -> None:
    """Render predicted-space Gaussians by folding unprojection into the camera.

    Note: `sharp predict` only uses this for SBS image output; video trajectories
    are rendered in world space via render_gaussians. Video output here is
    library API surface.
    """
    if metrics:
        metrics.inc("render_calls")
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px

    if params is None:
        params = camera.TrajectoryParams()

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering a checkpoint requires CUDA.")

    device = torch.device("cuda")
    gaussians = prune_gaussians(
        gaussians,
        min_opacity=min_opacity,
        min_scale=min_scale,
        max_splats=max_splats,
        score=prune_score,
    )
    gaussians_device = gaussians.to(device)

    intrinsics = _pinhole_intrinsics(f_px, width, height, device)
    u_pred_to_world = unprojection_matrix.to(device=device, dtype=torch.float32)

    camera_stats_scene = build_camera_stats_scene(gaussians_device, u_pred_to_world)
    camera_model = camera.create_camera_model(
        camera_stats_scene, intrinsics, resolution_px=metadata.resolution_px
    )

    # Number of camera animation loops.
    render_params = dataclasses.replace(params, num_repeats=3)
    if sbs_image_path is not None:
        render_params = dataclasses.replace(render_params, type="static")
        sbs_image_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = camera.create_eye_trajectory(
        camera_stats_scene, render_params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)

    _render_stereo_outputs(
        gaussians_device=gaussians_device,
        camera_model=camera_model,
        trajectory=trajectory,
        extrinsics_post=u_pred_to_world,
        renderer=renderer,
        device=device,
        output_path=output_path,
        sbs_image_path=sbs_image_path,
        sbs_image_format=sbs_image_format,
        sbs_jpeg_quality=sbs_jpeg_quality,
        sbs_image_frame=sbs_image_frame,
        align_crop=align_crop,
        stereo_baseline=stereo_baseline,
        stereo_mode=stereo_mode,
        stereo_convergence_depth=stereo_convergence_depth,
        stereo_convergence_norm=stereo_convergence_norm,
        metrics=metrics,
        sbs_async_writer=sbs_async_writer,
    )
