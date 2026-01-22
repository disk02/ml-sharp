"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import io as py_io
import logging
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import click
import numpy as np
import torch
import torch.utils.data
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

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


_SBS_ASYNC_QUEUE_SIZE = 8
_SBS_ASYNC_WORKERS = 1


class AsyncImageWriter:
    def __init__(self, maxsize: int = _SBS_ASYNC_QUEUE_SIZE, workers: int = _SBS_ASYNC_WORKERS):
        self._queue: queue.Queue[tuple[Path, np.ndarray | Image.Image, str | None, int] | None]
        self._queue = queue.Queue(maxsize=maxsize)
        self._threads: list[threading.Thread] = []
        self._error: BaseException | None = None
        self._closed = False
        for idx in range(workers):
            thread = threading.Thread(target=self._worker, name=f"sbs-writer-{idx}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def submit(
        self,
        path: Path,
        image: np.ndarray | Image.Image,
        *,
        sbs_image_format: str | None,
        sbs_jpeg_quality: int,
    ) -> None:
        self.raise_if_failed()
        if self._closed:
            raise click.ClickException("Async SBS writer is closed.")
        self._queue.put((path, image, sbs_image_format, sbs_jpeg_quality))
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise click.ClickException(f"Async SBS writer failed: {self._error}") from self._error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join()
        self.raise_if_failed()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            path, image, sbs_image_format, sbs_jpeg_quality = item
            try:
                self._save_image(path, image, sbs_image_format, sbs_jpeg_quality)
            except BaseException as exc:  # pragma: no cover - best-effort propagation
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    def _save_image(
        self,
        path: Path,
        image: np.ndarray | Image.Image,
        sbs_image_format: str | None,
        sbs_jpeg_quality: int,
    ) -> None:
        img = image if isinstance(image, Image.Image) else Image.fromarray(image, mode="RGB")
        img_format = Image.registered_extensions().get(path.suffix.lower())
        save_kwargs: dict[str, object] = {}
        if sbs_image_format == "jpg":
            img = img.convert("RGB")
            save_kwargs = {"quality": sbs_jpeg_quality, "subsampling": 0, "optimize": False}
        if img_format is None:
            img.save(path, **save_kwargs)
        else:
            bytes_io = py_io.BytesIO()
            img.save(bytes_io, format=img_format, **save_kwargs)
            with path.open("wb") as file_handle:
                file_handle.write(bytes_io.getvalue())


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
        "If provided without a value, saves to <output-path>/<image_stem>_sbs.jpg. "
        "If a directory path is provided, saves <dir>/<image_stem>_sbs.jpg."
    ),
)
@click.option(
    "--sbs-format",
    type=click.Choice(["jpg", "png"], case_sensitive=False),
    default=None,
    show_default=False,
    help="SBS image format when using --sbs-image (jpg or png).",
)
@click.option(
    "--sbs-jpeg-quality",
    type=click.IntRange(1, 100),
    default=90,
    show_default=True,
    help="JPEG quality (1-100) for --sbs-format=jpg.",
)
@click.option(
    "--png-format",
    is_flag=True,
    default=False,
    help="Select legacy PNG output for --sbs-image (equivalent to --sbs-format=png).",
)
@click.option(
    "--sbs-image-frame",
    type=int,
    default=0,
    show_default=True,
    help="Which frame index to save for --sbs-image.",
)
@click.option(
    "--stereo-strength",
    type=float,
    default=0.065,
    show_default=True,
    help=(
        "Absolute stereo baseline for SBS rendering (world units, default 0.065). "
        "Only used with --sbs-image."
    ),
)
@click.option(
    "--fast-preview-render",
    is_flag=True,
    default=False,
    help="Render SBS preview using predicted-space gaussians (skips world conversion).",
)
@click.option(
    "--tiling",
    is_flag=True,
    default=False,
    help="Enable tiled inference mode (requires --sbs-image and --fast-preview-render).",
)
@click.option(
    "--tile-size",
    type=int,
    default=1536,
    show_default=True,
    help="Nominal tile size in pixels (square tiles assumed).",
)
@click.option(
    "--tile-overlap",
    type=float,
    default=0.25,
    show_default=True,
    help="Fractional overlap between tiles in [0.0, 0.5).",
)
@click.option(
    "--tile-keep",
    type=float,
    default=None,
    show_default=True,
    help="Optional keep region fraction for tiles (defaults to derive from overlap).",
)
@click.option(
    "--sbs-min-opacity",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Minimum opacity for SBS splats. Suggested starting values: 0.005 or 0.01. "
        "Only used with --sbs-image."
    ),
)
@click.option(
    "--sbs-min-scale",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Minimum splat scale for SBS pruning (model-dependent units). Suggested starting "
        "value: 0.001. Only used with --sbs-image."
    ),
)
@click.option(
    "--sbs-max-splats",
    type=int,
    default=None,
    show_default=True,
    help="Optional cap on SBS splat count for faster preview rendering. Only used with --sbs-image.",
)
@click.option(
    "--sbs-prune-score",
    type=click.Choice(["opacity", "opacity_scale"], case_sensitive=False),
    default="opacity_scale",
    show_default=True,
    help="Score used to pick top-K SBS splats (opacity or opacity_scale). Only used with --sbs-image.",
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
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Micro-batch size for model forward (prediction only). "
        "Rendering/saving is still per-image."
    ),
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    sbs_image: Path | None,
    sbs_format: str | None,
    sbs_jpeg_quality: int,
    png_format: bool,
    sbs_image_frame: int,
    stereo_strength: float,
    fast_preview_render: bool,
    tiling: bool,
    tile_size: int,
    tile_overlap: float,
    tile_keep: float | None,
    sbs_min_opacity: float,
    sbs_min_scale: float,
    sbs_max_splats: int | None,
    sbs_prune_score: str,
    align_crop: bool,
    fast_preview_compare: bool,
    save_ply: bool | None,
    skip_world_conversion: bool,
    defer_world_conversion_for_export: bool,
    export_fp32: bool,
    device: str,
    amp: bool | None,
    amp_dtype: str,
    batch_size: int,
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
    if batch_size < 1:
        raise click.ClickException("--batch-size must be >= 1.")
    if tiling and (sbs_image is None or not fast_preview_render):
        raise click.ClickException(
            "--tiling is only supported with --sbs-image and --fast-preview-render."
        )
    if tile_overlap < 0.0 or tile_overlap >= 0.5:
        raise click.ClickException("--tile-overlap must be in [0.0, 0.5).")
    if tile_keep is not None and (tile_keep <= 0.0 or tile_keep > 1.0):
        raise click.ClickException("--tile-keep must be in (0.0, 1.0].")

    def _natural_sort_key(path: Path) -> list[tuple[int, object]]:
        relative_path = path.relative_to(input_path).as_posix()
        parts = re.split(r"(\d+)", relative_path)
        key: list[tuple[int, object]] = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append((0, int(part)))
            else:
                key.append((1, part.casefold()))
        return key

    # Ensure deterministic traversal order across filesystems before processing.
    image_paths.sort(key=_natural_sort_key)

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
    amp_dtype_to_use: torch.dtype | None = None
    if amp and device == "cuda":
        amp_dtype_to_use = amp_autocast_dtype
        if amp_autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            amp_dtype_to_use = torch.float16
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
    want_render_trajectory = with_rendering
    if not want_sbs_image and (sbs_format is not None or png_format):
        LOGGER.warning("--sbs-format/--png-format ignored because --sbs-image was not set.")
    effective_sbs_format = None
    if want_sbs_image:
        if sbs_format is None:
            if png_format:
                effective_sbs_format = "png"
            else:
                effective_sbs_format = "jpg"
        else:
            if png_format:
                LOGGER.warning("--png-format ignored because --sbs-format was provided.")
            effective_sbs_format = sbs_format.lower()
        if png_format and effective_sbs_format == "png":
            LOGGER.info("Using legacy PNG output via --png-format.")
        if effective_sbs_format == "jpg":
            LOGGER.info("SBS image format: jpg (quality %d).", sbs_jpeg_quality)
        elif effective_sbs_format == "png":
            LOGGER.info("SBS image format: png.")
    if fast_preview_render and not want_sbs_image:
        LOGGER.warning("--fast-preview-render is only used with --sbs-image. Disabling.")
        fast_preview_render = False
    if fast_preview_compare and not want_sbs_image:
        raise click.ClickException("--fast-preview-compare requires --sbs-image.")
    if fast_preview_compare and not fast_preview_render:
        raise click.ClickException("--fast-preview-compare requires --fast-preview-render.")
    if fast_preview_render and want_render_trajectory:
        raise click.ClickException(
            "--fast-preview-render is only supported for --sbs-image (not --render)."
        )
    if save_ply is None and want_sbs_image:
        effective_save_ply = False
    elif save_ply is None and not want_sbs_image:
        effective_save_ply = True
    else:
        effective_save_ply = bool(save_ply)
    if want_render_trajectory or want_sbs_image or fast_preview_render or fast_preview_compare:
        metrics.render_timing = RenderTiming()

    sbs_async_writer: AsyncImageWriter | None = None
    if want_sbs_image and not fast_preview_compare:
        sbs_async_writer = AsyncImageWriter()
        LOGGER.info("Async SBS image writer enabled.")

    def _finalize_prediction(
        *,
        prediction: PredictionResult,
        image_path: Path,
        rel_path: Path,
        out_dir: Path,
        image: np.ndarray,
        f_px: float,
        height: int,
        width: int,
        intrinsics: torch.Tensor,
        image_start: float,
        sbs_async_writer: AsyncImageWriter | None,
    ) -> None:
        # Determine SBS image output path (optional)
        sbs_image_path: Path | None = None
        if want_sbs_image:
            desired_suffix = ".png" if effective_sbs_format == "png" else ".jpg"
            sbs_out = Path(sbs_image)
            # Support `--sbs-image` with no value: default to output directory
            if sbs_out.name == "__AUTO__":
                sbs_image_path = out_dir / f"{image_path.stem}_sbs{desired_suffix}"
            else:
                # If a directory path is provided (no suffix), place an image per input
                if sbs_out.suffix == "":
                    sbs_base = sbs_out
                    if input_is_dir:
                        sbs_base = sbs_out / rel_path.parent
                        sbs_base.mkdir(parents=True, exist_ok=True)
                    sbs_image_path = sbs_base / f"{image_path.stem}_sbs{desired_suffix}"
                else:
                    # If multiple inputs are processed and a single file path is given, avoid overwrites
                    if len(image_paths) > 1:
                        rel_stem = "_".join(rel_path.with_suffix("").parts)
                        sbs_image_path = sbs_out.with_name(
                            f"{sbs_out.stem}_{rel_stem}{desired_suffix}"
                        )
                    else:
                        sbs_image_path = sbs_out.with_suffix(desired_suffix)
            if sbs_image_path is not None:
                sbs_image_path.parent.mkdir(parents=True, exist_ok=True)

        if want_render_trajectory or sbs_image_path is not None:
            if want_render_trajectory:
                output_video_path = (out_dir / image_path.stem).with_suffix(".mp4")
                LOGGER.info("Rendering trajectory to %s", output_video_path)
            else:
                # Placeholder path; render_gaussians will not write video when sbs_image_path is set.
                output_video_path = (out_dir / image_path.stem).with_suffix(".mp4")

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")

            render_start = perf_counter()
            sbs_prune_min_opacity = sbs_min_opacity if want_sbs_image else 0.0
            sbs_prune_min_scale = sbs_min_scale if want_sbs_image else 0.0
            sbs_prune_max_splats = sbs_max_splats if want_sbs_image else None
            normalized_prune_score = sbs_prune_score.lower()
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
                        sbs_image_format=effective_sbs_format,
                        sbs_jpeg_quality=sbs_jpeg_quality,
                        sbs_image_frame=sbs_image_frame,
                        align_crop=align_crop,
                        metrics=metrics,
                        sbs_async_writer=None,
                        stereo_baseline=stereo_strength,
                        min_opacity=sbs_prune_min_opacity,
                        min_scale=sbs_prune_min_scale,
                        max_splats=sbs_prune_max_splats,
                        prune_score=normalized_prune_score,
                    )
                render_gaussians_pred_space(
                    gaussians=prediction.pred,
                    metadata=metadata,
                    output_path=output_video_path,
                    unprojection_matrix=prediction.unprojection_matrix,
                    sbs_image_path=sbs_image_path,
                    sbs_image_format=effective_sbs_format,
                    sbs_jpeg_quality=sbs_jpeg_quality,
                    sbs_image_frame=sbs_image_frame,
                    align_crop=align_crop,
                    metrics=metrics,
                    sbs_async_writer=None if fast_preview_compare else sbs_async_writer,
                    stereo_baseline=stereo_strength,
                    min_opacity=sbs_prune_min_opacity,
                    min_scale=sbs_prune_min_scale,
                    max_splats=sbs_prune_max_splats,
                    prune_score=normalized_prune_score,
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
                    sbs_image_format=effective_sbs_format,
                    sbs_jpeg_quality=sbs_jpeg_quality,
                    sbs_image_frame=sbs_image_frame,
                    align_crop=align_crop,
                    metrics=metrics,
                    sbs_async_writer=sbs_async_writer,
                    stereo_baseline=stereo_strength if sbs_image_path is not None else 0.065,
                    min_opacity=sbs_prune_min_opacity,
                    min_scale=sbs_prune_min_scale,
                    max_splats=sbs_prune_max_splats,
                    prune_score=normalized_prune_score,
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

    run_start = perf_counter()
    try:
        if batch_size <= 1 or len(image_paths) == 1:
            for index, image_path in enumerate(image_paths, start=1):
                image_start = perf_counter()
                rel_path = image_path.relative_to(input_path) if input_is_dir else Path(image_path.name)
                out_dir = output_path / rel_path.parent if input_is_dir else output_path
                out_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info("Processing %s (%d/%d)", image_path, index, len(image_paths))
                io_start = perf_counter()
                try:
                    image, _, f_px = io.load_rgb(image_path)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    LOGGER.warning("Skipping unreadable image %s: %s", image_path, exc)
                    continue
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
                    want_render_trajectory
                    or (want_sbs_image and not fast_preview_render)
                    or fast_preview_compare
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
                    amp_dtype=amp_dtype_to_use,
                    metrics=metrics,
                    return_world=want_world_for_predict,
                    return_unprojection=fast_preview_render,
                    return_unprojection_context=fast_preview_render or effective_save_ply,
                )
                metrics.add_time("predict_total", perf_counter() - predict_start)
                _finalize_prediction(
                    prediction=prediction,
                    image_path=image_path,
                    rel_path=rel_path,
                    out_dir=out_dir,
                    image=image,
                    f_px=f_px,
                    height=height,
                    width=width,
                    intrinsics=intrinsics,
                    image_start=image_start,
                    sbs_async_writer=sbs_async_writer,
                )
        else:
            total_images = len(image_paths)
            for batch_start in range(0, total_images, batch_size):
                batch_paths = image_paths[batch_start : batch_start + batch_size]
                batch_items: list[dict[str, Any]] = []
                for offset, image_path in enumerate(batch_paths):
                    index = batch_start + offset + 1
                    image_start = perf_counter()
                    rel_path = (
                        image_path.relative_to(input_path) if input_is_dir else Path(image_path.name)
                    )
                    out_dir = output_path / rel_path.parent if input_is_dir else output_path
                    out_dir.mkdir(parents=True, exist_ok=True)
                    LOGGER.info("Processing %s (%d/%d)", image_path, index, total_images)
                    io_start = perf_counter()
                    try:
                        image, _, f_px = io.load_rgb(image_path)
                    except (OSError, UnidentifiedImageError, ValueError) as exc:
                        LOGGER.warning("Skipping unreadable image %s: %s", image_path, exc)
                        continue
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
                        want_render_trajectory
                        or (want_sbs_image and not fast_preview_render)
                        or fast_preview_compare
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
                            LOGGER.info(
                                "Using fast preview render; deferring world conversion for export."
                            )
                        else:
                            LOGGER.info("Using fast preview render; skipping world conversion.")
                    space: GaussianSpace = "world" if want_world_for_predict else "pred"
                    LOGGER.info(
                        "Computing world-space gaussians: %s", "yes" if space == "world" else "no"
                    )

                    preprocess_start = perf_counter() if metrics else None
                    image_resized_pt, disparity_factor, aux = preprocess_one(
                        image,
                        f_px,
                        torch.device(device),
                        target_size_wh=(1536, 1536),
                        dtype=torch.float32,
                    )
                    aux["metrics"] = metrics
                    preprocess_elapsed = 0.0
                    if metrics and preprocess_start is not None:
                        preprocess_elapsed = perf_counter() - preprocess_start
                        metrics.add_time("preprocess", preprocess_elapsed)
                    batch_items.append(
                        {
                            "image_path": image_path,
                            "rel_path": rel_path,
                            "out_dir": out_dir,
                            "image": image,
                            "f_px": f_px,
                            "height": height,
                            "width": width,
                            "intrinsics": intrinsics,
                            "image_start": image_start,
                            "preprocess_elapsed": preprocess_elapsed,
                            "want_world_for_predict": want_world_for_predict,
                            "aux": aux,
                            "image_resized_pt": image_resized_pt,
                            "disparity_factor": disparity_factor,
                        }
                    )

                if not batch_items:
                    continue
                image_resized_batch = torch.cat(
                    [item["image_resized_pt"] for item in batch_items], dim=0
                )
                disparity_factor_batch = torch.cat(
                    [item["disparity_factor"] for item in batch_items], dim=0
                )

                forward_start = perf_counter() if metrics else None
                try:
                    gaussians_ndc_batch = model_forward_batch(
                        gaussian_predictor,
                        image_resized_batch,
                        disparity_factor_batch,
                        amp_dtype=amp_dtype_to_use,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    raise click.ClickException(
                        f"CUDA out of memory during batched forward (batch_size={len(batch_items)}). "
                        "Try a smaller --batch-size."
                    ) from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        raise click.ClickException(
                            f"CUDA out of memory during batched forward (batch_size={len(batch_items)}). "
                            "Try a smaller --batch-size."
                        ) from exc
                    raise
                forward_elapsed = 0.0
                if metrics and forward_start is not None:
                    forward_elapsed = perf_counter() - forward_start
                    metrics.add_time("model_forward", forward_elapsed)

                postprocess_start = perf_counter() if metrics else None
                for batch_index, item in enumerate(batch_items):
                    gaussians_ndc_one = _slice_gaussians(gaussians_ndc_batch, batch_index)
                    prediction = postprocess_one(
                        gaussians_ndc_one,
                        item["aux"],
                        return_world=item["want_world_for_predict"],
                        return_unprojection=fast_preview_render or effective_save_ply,
                        device=torch.device(device),
                    )
                    item["prediction"] = prediction
                postprocess_elapsed = 0.0
                if metrics and postprocess_start is not None:
                    postprocess_elapsed = perf_counter() - postprocess_start
                    metrics.add_time("postprocess", postprocess_elapsed)

                batch_count = len(batch_items)
                forward_share = forward_elapsed / batch_count if batch_count else 0.0
                postprocess_share = postprocess_elapsed / batch_count if batch_count else 0.0
                for item in batch_items:
                    predict_total = (
                        item["preprocess_elapsed"] + forward_share + postprocess_share
                    )
                    metrics.add_time("predict_total", predict_total)
                    _finalize_prediction(
                        prediction=item["prediction"],
                        image_path=item["image_path"],
                        rel_path=item["rel_path"],
                        out_dir=item["out_dir"],
                        image=item["image"],
                        f_px=item["f_px"],
                        height=item["height"],
                        width=item["width"],
                        intrinsics=item["intrinsics"],
                        image_start=item["image_start"],
                        sbs_async_writer=sbs_async_writer,
                    )
    finally:
        if sbs_async_writer is not None:
            drain_start = perf_counter()
            sbs_async_writer.close()
            metrics.add_time("sbs_async_drain", perf_counter() - drain_start)
    metrics.add_time("run_total", perf_counter() - run_start)
    _log_metrics_summary(metrics)


def preprocess_one(
    image_np: np.ndarray,
    f_px: float,
    device: torch.device,
    *,
    target_size_wh: tuple[int, int],
    dtype: torch.dtype,
    reference_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    target_w, target_h = target_size_wh
    image_np = np.ascontiguousarray(image_np)
    if not image_np.flags.writeable:
        image_np = image_np.copy()
    image_pt = (
        torch.from_numpy(image_np)
        .permute(2, 0, 1)
        .contiguous()
        .to(device=device, dtype=dtype)
    )
    image_pt = image_pt / 255.0
    _, height, width = image_pt.shape
    width_for_disparity = width
    if reference_width is not None:
        if not isinstance(reference_width, int) or reference_width <= 0:
            raise click.ClickException("--reference_width must be a positive integer.")
        width_for_disparity = reference_width
    disparity_factor_pt = torch.tensor([f_px / width_for_disparity], dtype=dtype, device=device)
    image_resized_pt = TF.resize(
        image_pt,
        [target_h, target_w],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).unsqueeze(0)
    # target_size_wh is (W, H); resize expects (H, W).
    # image_resized_pt: (B, C, H, W) with B=1 for now.
    if __debug__:
        assert image_resized_pt.ndim == 4
        assert image_resized_pt.shape[0] == 1
        assert image_resized_pt.shape[1] == 3
        assert image_resized_pt.shape[2] == target_h
        assert image_resized_pt.shape[3] == target_w
        assert disparity_factor_pt.ndim >= 1
        assert disparity_factor_pt.shape[0] == 1
    aux = {
        "height": height,
        "width": width,
        "f_px": f_px,
        "target_w": target_w,
        "target_h": target_h,
    }
    return image_resized_pt, disparity_factor_pt, aux


def _slice_gaussians(gaussians: Gaussians3D, idx: int) -> Gaussians3D:
    """Return a view of a single batch element while preserving the batch dim."""
    sl = slice(idx, idx + 1)
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[sl],
        singular_values=gaussians.singular_values[sl],
        quaternions=gaussians.quaternions[sl],
        colors=gaussians.colors[sl],
        opacities=gaussians.opacities[sl],
    )


def model_forward_batch(
    predictor: torch.nn.Module,
    image_resized_pt: torch.Tensor,
    disparity_factor_pt: torch.Tensor,
    *,
    amp_dtype: torch.dtype | None,
) -> Any:
    with torch.inference_mode():
        # CUDA uses autocast when amp_dtype is set; CPU/MPS run without autocast.
        if image_resized_pt.device.type == "cuda":
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                return predictor(image_resized_pt, disparity_factor_pt)
        return predictor(image_resized_pt, disparity_factor_pt)


def postprocess_one(
    gaussians_ndc_one: Gaussians3D,
    aux: dict,
    *,
    return_world: bool,
    return_unprojection: bool,
    device: torch.device,
) -> PredictionResult:
    need_cast = any(
        tensor.dtype != torch.float32
        for tensor in (
            gaussians_ndc_one.mean_vectors,
            gaussians_ndc_one.singular_values,
            gaussians_ndc_one.quaternions,
            gaussians_ndc_one.colors,
            gaussians_ndc_one.opacities,
        )
    )
    if need_cast:
        mean_vectors = gaussians_ndc_one.mean_vectors.float()
        singular_values = gaussians_ndc_one.singular_values.float()
        quaternions = gaussians_ndc_one.quaternions.float()
        colors = gaussians_ndc_one.colors.float()
        opacities = gaussians_ndc_one.opacities.float()
    else:
        mean_vectors = gaussians_ndc_one.mean_vectors
        singular_values = gaussians_ndc_one.singular_values
        quaternions = gaussians_ndc_one.quaternions
        colors = gaussians_ndc_one.colors
        opacities = gaussians_ndc_one.opacities
    quat_norm = quaternions.norm(dim=-1, keepdim=True)
    quat_norm_abs = quat_norm.abs()
    quat_finite = torch.isfinite(quat_norm)
    quat_too_small = quat_norm_abs <= 1e-8
    quat_off_unit = (quat_norm_abs - 1.0).abs() > 1e-3
    quat_valid = quat_finite & ~quat_too_small
    need_quat_fix = bool((~quat_finite).any() | quat_too_small.any() | quat_off_unit.any())
    bad_scales = ~torch.isfinite(singular_values) | (singular_values <= 0)
    bad_scales_any = bool(bad_scales.any())
    if bad_scales_any:
        LOGGER.warning(
            "Repairing %d invalid singular value entries.",
            int(bad_scales.sum().item()),
        )
        singular_values = singular_values.clone()
        singular_values[bad_scales] = 1e-3
    if need_quat_fix:
        identity_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=quaternions.device, dtype=quaternions.dtype
        )
        quaternions = torch.where(
            quat_valid, quaternions / quat_norm.clamp_min(1e-8), identity_quat
        )

    if need_cast or need_quat_fix or bad_scales_any:
        gaussians_ndc = Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=singular_values,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
        )
    else:
        gaussians_ndc = gaussians_ndc_one

    gaussians_world = None
    unprojection_matrix = None
    unprojection_context = None
    if return_world or return_unprojection:
        height = aux["height"]
        width = aux["width"]
        f_px = aux["f_px"]
        target_w = aux["target_w"]
        target_h = aux["target_h"]
        intrinsics = (
            torch.tensor(
                [
                    [f_px, 0, (width - 1) / 2.0, 0],
                    [0, f_px, (height - 1) / 2.0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .to(device)
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= target_w / width
        intrinsics_resized[1] *= target_h / height
        unprojection_context = UnprojectionContext(
            intrinsics_resized=intrinsics_resized,
            internal_shape=(target_w, target_h),
            device=device,
        )
        if return_unprojection:
            unprojection_matrix = get_unprojection_matrix(
                torch.eye(4).to(device), intrinsics_resized, (target_w, target_h)
            )
        if return_world:
            gaussians_world = unproject_gaussians(
                gaussians_ndc,
                torch.eye(4).to(device),
                intrinsics_resized,
                (target_w, target_h),
                metrics=aux.get("metrics"),
            )

    return PredictionResult(
        pred=gaussians_ndc,
        world=gaussians_world,
        unprojection_matrix=unprojection_matrix,
        unprojection_context=unprojection_context,
    )


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
    target_size_wh = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    preprocess_start = perf_counter() if metrics else None
    image_resized_pt, disparity_factor, aux = preprocess_one(
        image,
        f_px,
        device,
        target_size_wh=target_size_wh,
        dtype=torch.float32,
    )
    aux["metrics"] = metrics
    if metrics and preprocess_start is not None:
        metrics.add_time("preprocess", perf_counter() - preprocess_start)

    LOGGER.info("Running inference.")
    forward_start = perf_counter() if metrics else None
    # amp_dtype is expected to be resolved by predict_cli (including bf16 fallback).
    amp_dtype_to_use = amp_dtype if amp_enabled else None
    gaussians_ndc_batch = model_forward_batch(
        predictor,
        image_resized_pt,
        disparity_factor,
        amp_dtype=amp_dtype_to_use,
    )
    if metrics and forward_start is not None:
        metrics.add_time("model_forward", perf_counter() - forward_start)

    LOGGER.info("Running postprocessing.")
    postprocess_start = perf_counter() if metrics else None
    gaussians_ndc_one = _slice_gaussians(gaussians_ndc_batch, 0)
    prediction = postprocess_one(
        gaussians_ndc_one,
        aux,
        return_world=return_world,
        return_unprojection=return_unprojection or return_unprojection_context,
        device=device,
    )
    if metrics and postprocess_start is not None:
        metrics.add_time("postprocess", perf_counter() - postprocess_start)

    return PredictionResult(
        pred=prediction.pred,
        world=prediction.world,
        unprojection_matrix=prediction.unprojection_matrix,
        unprojection_context=prediction.unprojection_context,
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
        "sbs_async_drain",
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
