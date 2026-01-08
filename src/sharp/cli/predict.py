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
    get_unprojection_matrix,
    save_ply,
    unproject_gaussians,
)
from sharp.utils.metrics import Metrics, RenderTiming

from sharp.rendering.gaussian_renderer import (
    render_gaussians,
    render_gaussians_pred_space,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
GaussianSpace = Literal["pred", "world"]


@dataclass(frozen=True)
class UnprojectionContext:
    intrinsics_resized: torch.Tensor
    internal_shape: tuple[int, int]
    device: torch.device


@dataclass(frozen=True)
class PredictionResult:
    pred: Gaussians3D
    world: Gaussians3D | None = None
    unprojection_matrix: torch.Tensor | None = None
    unprojection_context: UnprojectionContext | None = None


def _align_for_compare(
    baseline_img: np.ndarray, fast_img: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    def _normalize_channels(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return np.repeat(img[:, :, None], 3, axis=2)
        if img.ndim == 3 and img.shape[2] == 4:
            return img[:, :, :3]
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        raise click.ClickException(f"Unsupported image shape for compare: {img.shape}")

    baseline_img = _normalize_channels(baseline_img)
    fast_img = _normalize_channels(fast_img)

    if baseline_img.shape == fast_img.shape:
        return baseline_img, fast_img

    h = min(baseline_img.shape[0], fast_img.shape[0])
    w = min(baseline_img.shape[1], fast_img.shape[1])
    if h <= 0 or w <= 0:
        raise click.ClickException(
            f"Invalid compare crop size from shapes {baseline_img.shape} and {fast_img.shape}."
        )

    def _center_crop(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h0, w0 = img.shape[:2]
        top = max((h0 - target_h) // 2, 0)
        left = max((w0 - target_w) // 2, 0)
        return img[top : top + target_h, left : left + target_w]

    baseline_crop = _center_crop(baseline_img, h, w)
    fast_crop = _center_crop(fast_img, h, w)
    LOGGER.info(
        "Aligned compare shapes: baseline=%s fast=%s -> compared=(%d, %d, 3)",
        baseline_img.shape,
        fast_img.shape,
        h,
        w,
    )
    return baseline_crop, fast_crop


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
    "--fast-preview-render",
    is_flag=True,
    default=False,
    help="Render SBS preview using predicted-space gaussians (skips world conversion).",
)
@click.option(
    "--align-crop",
    is_flag=True,
    default=False,
    help="If set, auto-align and auto-crop the SBS stereo pair before saving the SBS image.",
)
@click.option(
    "--fast-preview-compare",
    is_flag=True,
    default=False,
    help="Compare fast preview render against baseline SBS rendering.",
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
    "--defer-world-conversion-for-export/--no-defer-world-conversion-for-export",
    "defer_world_conversion_for_export",
    default=False,
    show_default=True,
    help="Defer world-space conversion until export to keep preview responsive.",
)
@click.option(
    "--export-fp32",
    is_flag=True,
    default=False,
    help="Compute PLY export conversions in fp32 (even if inference uses AMP).",
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
    fast_preview_render: bool,
    align_crop: bool,
    fast_preview_compare: bool,
    save_ply: bool | None,
    skip_world_conversion: bool,
    defer_world_conversion_for_export: bool,
    export_fp32: bool,
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
    if fast_preview_render and not want_sbs_image:
        LOGGER.warning("--fast-preview-render is only used with --sbs-image. Disabling.")
        fast_preview_render = False
    if fast_preview_compare and not want_sbs_image:
        raise click.ClickException("--fast-preview-compare requires --sbs-image.")
    if fast_preview_compare and not fast_preview_render:
        raise click.ClickException("--fast-preview-compare requires --fast-preview-render.")
    if fast_preview_render and want_video:
        raise click.ClickException(
            "--fast-preview-render is only supported for --sbs-image (not --render)."
        )
    if save_ply is None and want_sbs_image:
        effective_save_ply = False
    elif save_ply is None and not want_sbs_image:
        effective_save_ply = True
    else:
        effective_save_ply = bool(save_ply)
    if want_video or want_sbs_image or fast_preview_render or fast_preview_compare:
        metrics.render_timing = RenderTiming()

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
        want_world_for_render = (
            want_video or (want_sbs_image and not fast_preview_render) or fast_preview_compare
        )
        defer_export_world = defer_world_conversion_for_export or (
            fast_preview_render and not fast_preview_compare
        )
        want_world_for_predict = want_world_for_render or (
            effective_save_ply and not defer_export_world
        )
        if skip_world_conversion and (want_world_for_predict or effective_save_ply):
            raise click.ClickException(
                "World-space conversion is required for rendering or PLY export. "
                "Disable --skip-world-conversion or disable those outputs."
            )
        if skip_world_conversion:
            LOGGER.info("Skipping world conversion (unproject/apply_transform).")
        if fast_preview_render and not want_world_for_predict:
            if effective_save_ply:
                LOGGER.info("Using fast preview render; deferring world conversion for export.")
            else:
                LOGGER.info("Using fast preview render; skipping world conversion.")
        space: GaussianSpace = "world" if want_world_for_predict else "pred"
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
            return_world=want_world_for_predict,
            return_unprojection=fast_preview_render,
            return_unprojection_context=fast_preview_render or effective_save_ply,
        )
        metrics.add_time("predict_total", perf_counter() - predict_start)

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

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")

            render_start = perf_counter()
            if fast_preview_render:
                if prediction.unprojection_matrix is None:
                    raise click.ClickException(
                        "Missing unprojection matrix for fast preview rendering."
                    )
                if fast_preview_compare:
                    if prediction.world is None:
                        raise click.ClickException(
                            "World-space Gaussians missing; compare requires world conversion."
                        )
                    if sbs_image_path is None:
                        raise click.ClickException(
                            "--fast-preview-compare requires --sbs-image."
                        )
                    baseline_path = sbs_image_path.with_name(
                        f"{sbs_image_path.stem}_baseline{sbs_image_path.suffix}"
                    )
                    render_gaussians(
                        gaussians=prediction.world,
                        metadata=metadata,
                        output_path=output_video_path,
                        sbs_image_path=baseline_path,
                        sbs_image_frame=sbs_image_frame,
                        align_crop=align_crop,
                        metrics=metrics,
                    )
                render_gaussians_pred_space(
                    gaussians=prediction.pred,
                    metadata=metadata,
                    output_path=output_video_path,
                    unprojection_matrix=prediction.unprojection_matrix,
                    sbs_image_path=sbs_image_path,
                    sbs_image_frame=sbs_image_frame,
                    align_crop=align_crop,
                    metrics=metrics,
                )
                if fast_preview_compare and sbs_image_path is not None:
                    try:
                        baseline_img = np.asarray(io.load_rgb(baseline_path)[0])
                        fast_img = np.asarray(io.load_rgb(sbs_image_path)[0])
                        baseline_img, fast_img = _align_for_compare(baseline_img, fast_img)
                        baseline_img = baseline_img.astype(np.float32) / 255.0
                        fast_img = fast_img.astype(np.float32) / 255.0
                        diff = np.abs(baseline_img - fast_img)
                        mae = float(diff.mean())
                        max_err = float(diff.max())
                        mse = float(np.mean(diff * diff))
                        psnr = float("inf") if mse == 0 else 20.0 * np.log10(1.0 / np.sqrt(mse))
                        LOGGER.info("Fast preview compare: MAE=%.6f Max=%.6f", mae, max_err)
                        LOGGER.info("Fast preview compare: PSNR=%.2f dB", psnr)
                    finally:
                        if baseline_path.exists():
                            baseline_path.unlink()
            else:
                if prediction.world is None:
                    raise click.ClickException(
                        "World-space Gaussians missing; rendering requires world conversion."
                    )
                render_gaussians(
                    gaussians=prediction.world,
                    metadata=metadata,
                    output_path=output_video_path,
                    sbs_image_path=sbs_image_path,
                    sbs_image_frame=sbs_image_frame,
                    align_crop=align_crop,
                    metrics=metrics,
                )
            metrics.add_time("render_total", perf_counter() - render_start)

        if effective_save_ply:
            world_gaussians = prediction.world
            if world_gaussians is None:
                LOGGER.info("Export requested: computing world-space gaussians for PLY.")
                if prediction.unprojection_context is None:
                    raise click.ClickException(
                        "World-space Gaussians missing; cannot export PLY without conversion."
                    )
                export_start = perf_counter()
                world_gaussians = _compute_world_gaussians_for_export(
                    prediction,
                    metrics=metrics,
                    export_fp32=export_fp32,
                )
                metrics.add_time("export_world_convert", perf_counter() - export_start)
                _log_export_fallbacks(metrics)
            else:
                LOGGER.info("Export requested: using cached world-space gaussians for PLY.")
            LOGGER.info("Saving 3DGS to %s", output_path)
            save_ply(
                world_gaussians,
                f_px,
                (height, width),
                out_dir / f"{image_path.stem}.ply",
                metrics=metrics,
            )
            LOGGER.info("Export complete.")
        else:
            if save_ply is None and want_sbs_image:
                LOGGER.info(
                    "Skipping .ply save because --sbs-image was requested (use --save-ply to override)."
                )
            else:
                LOGGER.info("Skipping .ply save because --no-save-ply was requested.")
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
    return_unprojection: bool = False,
    return_unprojection_context: bool = False,
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
    unprojection_matrix = None
    unprojection_context = None
    if return_world or return_unprojection or return_unprojection_context:
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
        unprojection_context = UnprojectionContext(
            intrinsics_resized=intrinsics_resized,
            internal_shape=internal_shape,
            device=device,
        )
        if return_unprojection:
            unprojection_matrix = get_unprojection_matrix(
                torch.eye(4).to(device), intrinsics_resized, internal_shape
            )
        if return_world:
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

    return PredictionResult(
        pred=gaussians_ndc,
        world=gaussians_world,
        unprojection_matrix=unprojection_matrix,
        unprojection_context=unprojection_context,
    )


def _cast_gaussians(gaussians: Gaussians3D, dtype: torch.dtype) -> Gaussians3D:
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors.to(dtype=dtype),
        singular_values=gaussians.singular_values.to(dtype=dtype),
        quaternions=gaussians.quaternions.to(dtype=dtype),
        colors=gaussians.colors.to(dtype=dtype),
        opacities=gaussians.opacities.to(dtype=dtype),
    )


def _compute_world_gaussians_for_export(
    prediction: PredictionResult,
    metrics: Metrics | None,
    export_fp32: bool,
) -> Gaussians3D:
    context = prediction.unprojection_context
    if context is None:
        raise click.ClickException("Missing unprojection context for export conversion.")
    pred_gaussians = prediction.pred
    if export_fp32:
        pred_gaussians = _cast_gaussians(pred_gaussians, torch.float32)
    extrinsics = torch.eye(4, device=context.device, dtype=context.intrinsics_resized.dtype)
    if context.device.type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            return unproject_gaussians(
                pred_gaussians,
                extrinsics,
                context.intrinsics_resized,
                context.internal_shape,
                metrics=metrics,
            )
    return unproject_gaussians(
        pred_gaussians,
        extrinsics,
        context.intrinsics_resized,
        context.internal_shape,
        metrics=metrics,
    )


def _log_export_fallbacks(metrics: Metrics | None) -> None:
    if metrics is None:
        return
    cov_nonfinite = metrics.counters.get("cov_nonfinite", 0)
    cpu_fallbacks = metrics.counters.get("eigh_cpu_fallback", 0)
    if cov_nonfinite > 0 or cpu_fallbacks > 0:
        LOGGER.warning(
            "Export covariance fallbacks: cov_nonfinite=%d eigh_cpu_fallback=%d",
            cov_nonfinite,
            cpu_fallbacks,
        )


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
        "export_world_convert",
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

    render_timing = metrics.render_timing
    if render_timing is not None:
        render_summary = render_timing.summarize()
        if render_summary:
            render_total = summary.get("render_total", {}).get("total", 0.0)
            header = (
                f"{'Stage':<30} {'mean(s)':>10} {'p50(s)':>10} {'p90(s)':>10} "
                f"{'total(s)':>10} {'%render':>10}"
            )
            lines = [header]
            for name in RenderTiming.stage_order + ["render_total_breakdown"]:
                stats = render_summary.get(name)
                if stats is None:
                    continue
                percent = (stats["total"] / render_total * 100.0) if render_total > 0 else 0.0
                lines.append(
                    f"{name:<30} "
                    f"{stats['mean']:>10.4f} "
                    f"{stats['p50']:>10.4f} "
                    f"{stats['p90']:>10.4f} "
                    f"{stats['total']:>10.4f} "
                    f"{percent:>10.2f}"
                )
            LOGGER.info(
                "Timing summary - render_total breakdown (seconds):\n%s", "\n".join(lines)
            )

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
