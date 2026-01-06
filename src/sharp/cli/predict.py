"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)
from sharp.utils.metrics import Metrics

from sharp.rendering.gaussian_renderer import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
GaussianSpace = Literal["pred", "world"]


@dataclass(frozen=True)
class PredictionResult:
    pred: Gaussians3D
    world: Gaussians3D | None = None


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
    "--skip-world-conversion",
    is_flag=True,
    default=False,
    help="Skip unproject/apply_transform world-space conversion (fast path).",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option(
    "--amp/--no-amp",
    "amp",
    default=None,
    show_default=False,
    help="Enable automatic mixed precision on CUDA (default: enabled on CUDA).",
)
@click.option(
    "--amp-dtype",
    type=click.Choice(["fp16", "bf16"], case_sensitive=False),
    default="fp16",
    show_default=True,
    help="Data type for CUDA AMP autocast.",
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
    skip_world_conversion: bool,
    device: str,
    amp: bool | None,
    amp_dtype: str,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths: list[Path] = []
    input_is_dir = input_path.is_dir()
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(path for path in input_path.rglob(f"*{ext}") if path.is_file())

    if len(image_paths) == 0:
        if input_is_dir:
            raise click.ClickException(f"No valid inputs found under {input_path}.")
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Input root: %s", input_path)
    LOGGER.info("Output root: %s", output_path)
    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if amp is None:
        amp = device == "cuda"
    elif amp and device != "cuda":
        LOGGER.warning("AMP is only supported on CUDA. Disabling AMP on %s.", device)
        amp = False

    amp_dtype_lower = amp_dtype.lower()
    amp_autocast_dtype = torch.float16 if amp_dtype_lower == "fp16" else torch.bfloat16
    LOGGER.info(
        "Predict settings: device=%s amp=%s amp_dtype=%s",
        device,
        "enabled" if amp else "disabled",
        amp_dtype_lower,
    )

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
    metrics = Metrics()

    want_sbs_image = sbs_image is not None
    want_video = with_rendering
    if save_ply is None and want_sbs_image:
        effective_save_ply = False
    elif save_ply is None and not want_sbs_image:
        effective_save_ply = True
    else:
        effective_save_ply = bool(save_ply)

    run_start = perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        image_start = perf_counter()
        rel_path = image_path.relative_to(input_path) if input_is_dir else Path(image_path.name)
        out_dir = output_path / rel_path.parent if input_is_dir else output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Processing %s (%d/%d)", image_path, index, len(image_paths))
        io_start = perf_counter()
        image, _, f_px = io.load_rgb(image_path)
        metrics.add_time("io_decode", perf_counter() - io_start)
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
        want_world = effective_save_ply or want_video or want_sbs_image
        if skip_world_conversion and want_world:
            raise click.ClickException(
                "World-space conversion is required for rendering or PLY export. "
                "Disable --skip-world-conversion or disable those outputs."
            )
        if skip_world_conversion:
            LOGGER.info("Skipping world conversion (unproject/apply_transform).")
        space: GaussianSpace = "world" if want_world else "pred"
        LOGGER.info("Computing world-space gaussians: %s", "yes" if space == "world" else "no")

        predict_start = perf_counter()
        prediction = predict_image(
            gaussian_predictor,
            image,
            f_px,
            torch.device(device),
            amp_enabled=amp,
            amp_dtype=amp_autocast_dtype,
            metrics=metrics,
            return_world=want_world,
        )
        metrics.add_time("predict_total", perf_counter() - predict_start)

        if effective_save_ply:
            LOGGER.info("Saving 3DGS to %s", output_path)
            if prediction.world is None:
                raise click.ClickException(
                    "World-space Gaussians missing; cannot export PLY without conversion."
                )
            save_ply(
                prediction.world,
                f_px,
                (height, width),
                out_dir / f"{image_path.stem}.ply",
                metrics=metrics,
            )
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
                sbs_image_path = out_dir / f"{image_path.stem}_sbs.png"
            else:
                # If a directory path is provided (no suffix), place an image per input
                if sbs_out.suffix == "":
                    sbs_base = sbs_out
                    if input_is_dir:
                        sbs_base = sbs_out / rel_path.parent
                        sbs_base.mkdir(parents=True, exist_ok=True)
                    sbs_image_path = sbs_base / f"{image_path.stem}_sbs.png"
                else:
                    # If multiple inputs are processed and a single file path is given, avoid overwrites
                    if len(image_paths) > 1:
                        rel_stem = "_".join(rel_path.with_suffix("").parts)
                        sbs_image_path = sbs_out.with_name(
                            f"{sbs_out.stem}_{rel_stem}{sbs_out.suffix}"
                        )
                    else:
                        sbs_image_path = sbs_out
            if sbs_image_path is not None:
                sbs_image_path.parent.mkdir(parents=True, exist_ok=True)

        if want_video or sbs_image_path is not None:
            if want_video:
                output_video_path = (out_dir / image_path.stem).with_suffix(".mp4")
                LOGGER.info("Rendering trajectory to %s", output_video_path)
            else:
                # Placeholder path; render_gaussians will not write video when sbs_image_path is set.
                output_video_path = (out_dir / image_path.stem).with_suffix(".mp4")

            if prediction.world is None:
                raise click.ClickException(
                    "World-space Gaussians missing; rendering requires world conversion."
                )
            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")

            render_start = perf_counter()
            render_gaussians(
                gaussians=prediction.world,
                metadata=metadata,
                output_path=output_video_path,
                sbs_image_path=sbs_image_path,
                sbs_image_frame=sbs_image_frame,
                metrics=metrics,
            )
            metrics.add_time("render_total", perf_counter() - render_start)
        metrics.add_time("per_image_total", perf_counter() - image_start)
    metrics.add_time("run_total", perf_counter() - run_start)
    _log_metrics_summary(metrics)


def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    metrics: Metrics | None = None,
    return_world: bool = True,
) -> PredictionResult:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    preprocess_start = perf_counter() if metrics else None
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    if metrics and preprocess_start is not None:
        metrics.add_time("preprocess", perf_counter() - preprocess_start)

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    forward_start = perf_counter() if metrics else None
    with torch.inference_mode():
        if amp_enabled and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                gaussians_ndc = predictor(image_resized_pt, disparity_factor)
        else:
            gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    if metrics and forward_start is not None:
        metrics.add_time("model_forward", perf_counter() - forward_start)

    LOGGER.info("Running postprocessing.")
    postprocess_start = perf_counter() if metrics else None
    gaussians_ndc = Gaussians3D(
        mean_vectors=gaussians_ndc.mean_vectors.float(),
        singular_values=gaussians_ndc.singular_values.float(),
        quaternions=gaussians_ndc.quaternions.float(),
        colors=gaussians_ndc.colors.float(),
        opacities=gaussians_ndc.opacities.float(),
    )
    quaternions = gaussians_ndc.quaternions
    quat_norm = quaternions.norm(dim=-1, keepdim=True)
    quat_valid = torch.isfinite(quat_norm) & (quat_norm > 1e-8)
    identity_quat = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=quaternions.device, dtype=quaternions.dtype
    )
    quaternions = torch.where(
        quat_valid, quaternions / quat_norm.clamp_min(1e-8), identity_quat
    )

    singular_values = gaussians_ndc.singular_values
    bad_scales = ~torch.isfinite(singular_values) | (singular_values <= 0)
    if bad_scales.any():
        LOGGER.warning(
            "Repairing %d invalid singular value entries.",
            int(bad_scales.sum().item()),
        )
        singular_values = singular_values.clone()
        singular_values[bad_scales] = 1e-3
    gaussians_ndc = Gaussians3D(
        mean_vectors=gaussians_ndc.mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians_ndc.colors,
        opacities=gaussians_ndc.opacities,
    )

    gaussians_world = None
    if return_world:
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
        gaussians_world = unproject_gaussians(
            gaussians_ndc,
            torch.eye(4).to(device),
            intrinsics_resized,
            internal_shape,
            metrics=metrics,
        )
    if metrics and postprocess_start is not None:
        metrics.add_time("postprocess", perf_counter() - postprocess_start)

    return PredictionResult(pred=gaussians_ndc, world=gaussians_world)


def _log_metrics_summary(metrics: Metrics) -> None:
    summary = metrics.summarize()
    if not summary:
        return

    stage_order = [
        "io_decode",
        "preprocess",
        "model_forward",
        "postprocess",
        "unproject_total",
        "apply_transform",
        "decompose_covariance",
        "predict_total",
        "render_total",
        "save_ply",
        "per_image_total",
        "run_total",
    ]

    header = f"{'Stage':<30} {'mean(s)':>10} {'p50(s)':>10} {'p90(s)':>10} {'total(s)':>10}"
    lines = [header]
    for name in stage_order:
        stats = summary.get(name)
        if stats is None:
            continue
        lines.append(
            f"{name:<30} "
            f"{stats['mean']:>10.4f} "
            f"{stats['p50']:>10.4f} "
            f"{stats['p90']:>10.4f} "
            f"{stats['total']:>10.4f}"
        )
    LOGGER.info("Timing summary (seconds):\n%s", "\n".join(lines))

    counter_order = [
        "cov_nonfinite",
        "eigh_retry_chunks",
        "eigh_cpu_fallback",
        "ortho_warn",
        "render_calls",
        "render_frames",
    ]
    counter_lines = ["Counters:"]
    for name in counter_order:
        counter_lines.append(f"  {name}={metrics.counters.get(name, 0)}")
    LOGGER.info("%s", "\n".join(counter_lines))
