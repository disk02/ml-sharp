"""Contains basic data structures and functionality for 3D Gaussians.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from time import perf_counter
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from sharp.utils import color_space as cs_utils
from sharp.utils import linalg
from sharp.utils.metrics import Metrics

LOGGER = logging.getLogger(__name__)


BackgroundColor = Literal["black", "white", "random_color", "random_pixel"]


class Gaussians3D(NamedTuple):
    """Represents a collection of 3D Gaussians."""

    mean_vectors: torch.Tensor
    singular_values: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor

    def to(self, device: torch.device) -> Gaussians3D:
        """Move Gaussians to device."""
        return Gaussians3D(
            mean_vectors=self.mean_vectors.to(device),
            singular_values=self.singular_values.to(device),
            quaternions=self.quaternions.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
        )


PruneScore = Literal["opacity", "opacity_scale"]


@torch.no_grad()
def prune_gaussians(
    gaussians: Gaussians3D,
    *,
    min_opacity: float = 0.0,
    min_scale: float = 0.0,
    max_splats: int | None = None,
    score: PruneScore = "opacity_scale",
) -> Gaussians3D:
    """Prune Gaussians by opacity/scale thresholds and optional top-K scoring.

    Batched inputs keep rectangular output by prioritizing splats that pass thresholds
    and filling remaining slots by score order.
    """
    if min_opacity == 0.0 and min_scale == 0.0 and max_splats is None:
        return gaussians
    if score not in ("opacity", "opacity_scale"):
        raise ValueError(f"Unsupported prune score: {score}")
    if max_splats is not None and max_splats < 1:
        raise ValueError("max_splats must be >= 1 when provided.")

    opacities = gaussians.opacities
    if opacities.ndim > 1 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    max_scale = gaussians.singular_values.max(dim=-1).values

    def _scores(opacity_vals: torch.Tensor, scale_vals: torch.Tensor) -> torch.Tensor:
        if score == "opacity":
            return opacity_vals
        return opacity_vals * scale_vals

    def _select_unbatched(
        gaussians_in: Gaussians3D, opacities_in: torch.Tensor, max_scale_in: torch.Tensor
    ) -> Gaussians3D:
        mask = (opacities_in >= min_opacity) & (max_scale_in >= min_scale)
        scores = _scores(opacities_in, max_scale_in)
        if mask.any():
            candidate_idx = mask.nonzero(as_tuple=False).flatten()
        else:
            LOGGER.warning(
                "All splats pruned; keeping the best-scoring splat to avoid empty render."
            )
            candidate_idx = scores.argmax().view(1)

        if max_splats is not None:
            k = min(max_splats, int(candidate_idx.numel()))
            topk = torch.topk(scores[candidate_idx], k=k).indices
            candidate_idx = candidate_idx[topk]

        def _index(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.index_select(0, candidate_idx)

        return Gaussians3D(
            mean_vectors=_index(gaussians_in.mean_vectors),
            singular_values=_index(gaussians_in.singular_values),
            quaternions=_index(gaussians_in.quaternions),
            colors=_index(gaussians_in.colors),
            opacities=_index(gaussians_in.opacities),
        )

    if gaussians.mean_vectors.ndim == 2:
        return _select_unbatched(gaussians, opacities, max_scale)
    if gaussians.mean_vectors.shape[0] == 1:
        squeezed = Gaussians3D(
            mean_vectors=gaussians.mean_vectors[0],
            singular_values=gaussians.singular_values[0],
            quaternions=gaussians.quaternions[0],
            colors=gaussians.colors[0],
            opacities=gaussians.opacities[0],
        )
        pruned = _select_unbatched(
            squeezed,
            opacities[0],
            max_scale[0],
        )
        return Gaussians3D(
            mean_vectors=pruned.mean_vectors.unsqueeze(0),
            singular_values=pruned.singular_values.unsqueeze(0),
            quaternions=pruned.quaternions.unsqueeze(0),
            colors=pruned.colors.unsqueeze(0),
            opacities=pruned.opacities.unsqueeze(0),
        )

    batch_size, num_splats = gaussians.mean_vectors.shape[:2]
    scores_batched = _scores(opacities, max_scale)
    target_count = num_splats
    if max_splats is not None:
        target_count = min(target_count, max_splats)
    target_count = max(target_count, 1)

    def _select_indices(batch: int) -> torch.Tensor:
        mask = (opacities[batch] >= min_opacity) & (max_scale[batch] >= min_scale)
        scores = scores_batched[batch]
        kept_idx = mask.nonzero(as_tuple=False).flatten()
        dropped_idx = (~mask).nonzero(as_tuple=False).flatten()
        if kept_idx.numel() == 0:
            LOGGER.warning(
                "All splats pruned for batch %d; keeping the best-scoring splat.",
                batch,
            )
        if kept_idx.numel():
            kept_order = torch.argsort(scores[kept_idx], descending=True)
        else:
            kept_order = kept_idx
        if dropped_idx.numel():
            drop_order = torch.argsort(scores[dropped_idx], descending=True)
        else:
            drop_order = dropped_idx
        ordered = torch.cat(
            (
                kept_idx[kept_order] if kept_idx.numel() else kept_idx,
                dropped_idx[drop_order] if dropped_idx.numel() else dropped_idx,
            ),
            dim=0,
        )
        if ordered.numel() == 0:
            ordered = scores.argmax().view(1)
        return ordered[:target_count]

    def _gather_field(tensor: torch.Tensor) -> torch.Tensor:
        gathered: list[torch.Tensor] = []
        for batch in range(batch_size):
            chosen = _select_indices(batch)
            gathered.append(tensor[batch].index_select(0, chosen))
        return torch.stack(gathered, dim=0)

    return Gaussians3D(
        mean_vectors=_gather_field(gaussians.mean_vectors),
        singular_values=_gather_field(gaussians.singular_values),
        quaternions=_gather_field(gaussians.quaternions),
        colors=_gather_field(gaussians.colors),
        opacities=_gather_field(gaussians.opacities),
    )


class SceneMetaData(NamedTuple):
    """Meta data about Gaussian scene."""

    focal_length_px: float
    resolution_px: tuple[int, int]
    color_space: cs_utils.ColorSpace


def get_unprojection_matrix(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Compute unprojection matrix to transform Gaussians to Euclidean space.

    Args:
        extrinsics: The 4x4 extrinsics matrix of the camera view.
        intrinsics: The 4x4 intrinsics matrix of the camera view.
        image_shape: The (width, height) of the input image.

    Returns:
        A 4x4 matrix to transform Gaussians from NDC space to Euclidean space.
    """
    device = intrinsics.device
    image_width, image_height = image_shape
    # This matrix converts OpenCV pixel coordinates to NDC coordinates where
    # (-1, 1) denotes the top left and (1, 1) the bottom right of the image.
    #
    # Note that premultiplying the intrinsics with ndc_matrix typically yields a matrix
    # that simply scales the x-axis by 2 * focal_length / image_width and the y-axis by
    # 2 * focal_length / image_height.
    ndc_matrix = torch.tensor(
        [
            [2.0 / image_width, 0.0, -1.0, 0.0],
            [0.0, 2.0 / image_height, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return torch.linalg.inv(ndc_matrix @ intrinsics @ extrinsics)


def unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    metrics: Metrics | None = None,
) -> Gaussians3D:
    """Unproject Gaussians from NDC space to world coordinates."""
    start_time = perf_counter() if metrics else None
    unprojection_matrix = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    gaussians = apply_transform(gaussians_ndc, unprojection_matrix[:3], metrics=metrics)
    if metrics and start_time is not None:
        metrics.add_time("unproject_total", perf_counter() - start_time)
    return gaussians


def apply_transform(
    gaussians: Gaussians3D, transform: torch.Tensor, metrics: Metrics | None = None
) -> Gaussians3D:
    """Apply an affine transformation to 3D Gaussians.

    Args:
        gaussians: The Gaussians to transform.
        transform: An affine transform with shape 3x4.

    Returns:
        The transformed Gaussians.

    Note: This operation is not differentiable.
    """
    start_time = perf_counter() if metrics else None
    transform_linear = transform[..., :3, :3]
    transform_offset = transform[..., :3, 3]

    mean_vectors = gaussians.mean_vectors @ transform_linear.T + transform_offset
    covariance_matrices = compose_covariance_matrices(
        gaussians.quaternions, gaussians.singular_values
    )
    covariance_matrices = (
        transform_linear @ covariance_matrices @ transform_linear.transpose(-1, -2)
    )
    quaternions, singular_values = decompose_covariance_matrices(
        covariance_matrices, metrics=metrics
    )
    if metrics and start_time is not None:
        metrics.add_time("apply_transform", perf_counter() - start_time)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def build_camera_stats_scene(
    gaussians: Gaussians3D,
    unprojection_matrix: torch.Tensor,
    sample_size: int = 16384,
) -> Gaussians3D:
    """Build a deterministic world-space subset of Gaussians for camera statistics.

    Camera statistics (depth quantiles, trajectory extents) depend only on splat
    mean positions. This transforms the means to world space and keeps at most
    ``sample_size`` points chosen with a fixed stride, so results are identical
    across runs without invoking any random number generators.

    Args:
        gaussians: The Gaussians (batched or unbatched) to build statistics for.
        unprojection_matrix: A 4x4 matrix transforming from NDC to world space.
        sample_size: The maximum number of points to include.

    Returns:
        An unbatched Gaussians3D containing world-space means for camera statistics.
    """
    mean_vectors = gaussians.mean_vectors
    singular_values = gaussians.singular_values
    quaternions = gaussians.quaternions
    colors = gaussians.colors
    opacities = gaussians.opacities
    if mean_vectors.ndim == 3:
        mean_vectors = mean_vectors[0]
        singular_values = singular_values[0]
        quaternions = quaternions[0]
        colors = colors[0]
        opacities = opacities[0]

    num_points = int(mean_vectors.shape[0])
    if num_points > sample_size:
        step = max(num_points // sample_size, 1)
        sample_idx = torch.arange(0, num_points, step, device=mean_vectors.device)
        sample_idx = sample_idx[:sample_size]
        mean_vectors = mean_vectors[sample_idx]
        singular_values = singular_values[sample_idx]
        quaternions = quaternions[sample_idx]
        colors = colors[sample_idx]
        opacities = opacities[sample_idx]

    transform_linear = unprojection_matrix[:3, :3]
    transform_offset = unprojection_matrix[:3, 3]
    mean_vectors = mean_vectors @ transform_linear.T + transform_offset

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def decompose_covariance_matrices(
    covariance_matrices: torch.Tensor,
    metrics: Metrics | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose 3D covariance matrices into quaternions and singular values.

    Args:
        covariance_matrices: The covariance matrices to decompose.

    Returns:
        Quaternion and singular values corresponding to the orientation and scales of
        the diagonalized matrix.

    Note: This operation is not differentiable.
    """
    start_time = perf_counter() if metrics else None
    device = covariance_matrices.device
    dtype = covariance_matrices.dtype

    # Use symmetric eigendecomposition on-device to avoid CPU round-trips.
    covariance_matrices = covariance_matrices.detach().to(torch.float32)
    covariance_matrices = 0.5 * (
        covariance_matrices + covariance_matrices.transpose(-1, -2)
    )
    finite_mask = torch.isfinite(covariance_matrices).all(dim=(-1, -2))

    flat_covariances = covariance_matrices.reshape(-1, 3, 3)
    flat_mask = finite_mask.reshape(-1)
    flat_evals = torch.empty(
        (flat_covariances.shape[0], 3),
        device=covariance_matrices.device,
        dtype=covariance_matrices.dtype,
    )
    flat_evecs = torch.empty(
        (flat_covariances.shape[0], 3, 3),
        device=covariance_matrices.device,
        dtype=covariance_matrices.dtype,
    )

    invalid_indices = (~flat_mask).nonzero(as_tuple=False).flatten()
    invalid_count = int(invalid_indices.numel())
    need_invalid_count = (
        metrics is not None
        or LOGGER.isEnabledFor(logging.DEBUG)
        or LOGGER.isEnabledFor(logging.WARNING)
    )
    num_invalid = invalid_count if need_invalid_count else None
    if num_invalid is not None and num_invalid > 0:
        LOGGER.warning(
            "Received %d non-finite covariance matrices. Falling back to CPU for them.",
            num_invalid,
        )
        if metrics:
            metrics.inc("cov_nonfinite", num_invalid)

    def _cpu_eigh(covariances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        evals, evecs = torch.linalg.eigh(covariances)
        return evals, evecs

    cpu_fallbacks = invalid_count

    eye3 = torch.eye(
        3, device=covariance_matrices.device, dtype=covariance_matrices.dtype
    )

    def _batched_eigh_chunked(
        indices: torch.Tensor, chunk_size: int, jitter: float | None = None
    ) -> tuple[torch.Tensor, int]:
        failed_chunks: list[torch.Tensor] = []
        for i in range(0, indices.numel(), chunk_size):
            chunk = indices[i : i + chunk_size]
            try:
                cov_chunk = flat_covariances[chunk]
                if jitter is not None:
                    cov_chunk = cov_chunk + jitter * eye3
                evals_chunk, evecs_chunk = torch.linalg.eigh(cov_chunk)
                flat_evals[chunk] = evals_chunk
                flat_evecs[chunk] = evecs_chunk
            except RuntimeError:
                failed_chunks.append(chunk)
        if not failed_chunks:
            return (
                torch.empty((0,), device=indices.device, dtype=indices.dtype),
                0,
            )
        return torch.cat(failed_chunks, dim=0), len(failed_chunks)

    valid_indices = flat_mask.nonzero(as_tuple=False).flatten()
    failed_indices = torch.empty(
        (0,), device=valid_indices.device, dtype=valid_indices.dtype
    )
    if valid_indices.numel() > 0:
        chunk_size = 1024
        LOGGER.debug(
            "Attempting batched EIGH on %d matrices with chunk_size=%d.",
            int(valid_indices.numel()),
            chunk_size,
        )
        failed_indices, failed_chunks = _batched_eigh_chunked(valid_indices, chunk_size)
        if failed_indices.numel() > 0:
            if metrics:
                metrics.inc("eigh_retry_chunks", failed_chunks)
            chunk_size = max(16, chunk_size // 4)
            LOGGER.warning(
                "Retrying batched EIGH on %d matrices with chunk_size=%d.",
                int(failed_indices.numel()),
                chunk_size,
            )
            failed_indices, _ = _batched_eigh_chunked(
                failed_indices, chunk_size, jitter=1e-8
            )

    if invalid_count > 0:
        evals_invalid, evecs_invalid = _cpu_eigh(
            flat_covariances[invalid_indices].cpu().to(torch.float64)
        )
        flat_evals[invalid_indices] = evals_invalid.to(
            device=device, dtype=covariance_matrices.dtype
        )
        flat_evecs[invalid_indices] = evecs_invalid.to(
            device=device, dtype=covariance_matrices.dtype
        )

    if failed_indices.numel() > 0:
        cpu_fallbacks += int(failed_indices.numel())
        evals_failed, evecs_failed = _cpu_eigh(
            flat_covariances[failed_indices].cpu().to(torch.float64)
        )
        flat_evals[failed_indices] = evals_failed.to(
            device=device, dtype=covariance_matrices.dtype
        )
        flat_evecs[failed_indices] = evecs_failed.to(
            device=device, dtype=covariance_matrices.dtype
        )

    if failed_indices.numel() > 0 or (num_invalid is not None and num_invalid > 0):
        LOGGER.warning(
            "Covariance decomposition fallbacks: failed_indices=%d cpu_fallbacks=%d",
            int(failed_indices.numel()),
            cpu_fallbacks,
        )
    if metrics and cpu_fallbacks > 0:
        metrics.inc("eigh_cpu_fallback", cpu_fallbacks)

    evals = flat_evals.reshape(*covariance_matrices.shape[:-1])
    evecs = flat_evecs.reshape(*covariance_matrices.shape)

    # Sort eigenvalues descending (largest scale first) and reorder eigenvectors (columns).
    sort_idx = evals.argsort(dim=-1, descending=True)
    evals = evals.gather(-1, sort_idx)
    evecs = evecs.gather(
        -1, sort_idx.unsqueeze(-2).expand(*sort_idx.shape[:-1], 3, 3)
    )

    # NOTE: it is possible that eigenvectors form a reflection. Correct to rotation.
    det = torch.linalg.det(evecs)
    reflection_mask = det < 0
    evecs = evecs.clone()
    evecs[..., :, -1] *= torch.where(reflection_mask[..., None], -1.0, 1.0).to(
        evecs.dtype
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        flat_mask = reflection_mask.reshape(-1)
        n = flat_mask.numel()
        k = min(4096, n)
        if k > 0:
            idx = torch.randint(0, n, (k,), device=flat_mask.device)
            sample_mask = flat_mask[idx]
            frac = sample_mask.float().mean().item()
            r_mats = evecs.reshape(-1, 3, 3)[idx]
            rtr = r_mats.transpose(-1, -2) @ r_mats
            identity = torch.eye(3, device=r_mats.device, dtype=r_mats.dtype)
            ortho_err = (rtr - identity).abs().max().item()
            LOGGER.debug(
                "EIGH diagnostics (sampled k=%d): reflection_frac=%.3f ortho_max=%.2e",
                k,
                frac,
                ortho_err,
            )
            if metrics:
                metrics.inc("reflection_sampled_k", k)
                metrics.inc("reflection_sampled_hits", int(sample_mask.sum().item()))
            if ortho_err > 1e-2:
                LOGGER.warning(
                    "EIGH orthonormality max|RtR-I| too high (sampled): %.2e", ortho_err
                )

    quaternions = linalg.quaternions_from_rotation_matrices(evecs)
    quaternions = quaternions.to(dtype=dtype, device=device)
    singular_values = torch.sqrt(evals.clamp_min(0.0)).to(dtype=dtype, device=device)
    if metrics and start_time is not None:
        metrics.add_time("decompose_covariance", perf_counter() - start_time)
    return quaternions, singular_values


def compose_covariance_matrices(
    quaternions: torch.Tensor, singular_values: torch.Tensor
) -> torch.Tensor:
    """Compose 3D covariance matrices into quaternions and singular values.

    Args:
        quaternions: The quaternions describing the principal basis.
        singular_values: The scales of the diagonalized matrix.

    Returns:
        The 3x3 covariances matrices.
    """
    device = quaternions.device
    rotations = linalg.rotation_matrices_from_quaternions(quaternions)
    diagonal_matrix = torch.eye(3, device=device) * singular_values[..., :, None]
    return rotations @ diagonal_matrix.square() @ rotations.transpose(-1, -2)


def convert_spherical_harmonics_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 spherical harmonics to RGB.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return sh0 * coeff_degree0 + 0.5


def convert_rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to degree-0 spherical harmonics.

    Reference:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    """
    coeff_degree0 = np.sqrt(1.0 / (4.0 * np.pi))
    return (rgb - 0.5) / coeff_degree0


def load_ply(path: Path) -> tuple[Gaussians3D, SceneMetaData]:
    """Loads a ply from a file."""
    plydata = PlyData.read(path)

    vertices = next(filter(lambda x: x.name == "vertex", plydata.elements))

    properties = ["x", "y", "z"]
    properties.extend([f"f_dc_{i}" for i in range(3)])
    properties.extend([f"scale_{i}" for i in range(3)])
    properties.extend([f"rot_{i}" for i in range(3)])

    for prop in properties:
        if prop not in vertices:
            raise KeyError(f"Incompatible ply file: property {prop} not found in ply elements.")
    mean_vectors = np.stack(
        (
            np.asarray(vertices["x"]),
            np.asarray(vertices["y"]),
            np.asarray(vertices["z"]),
        ),
        axis=1,
    )

    scale_logits = np.stack(
        (
            np.asarray(vertices["scale_0"]),
            np.asarray(vertices["scale_1"]),
            np.asarray(vertices["scale_2"]),
        ),
        axis=1,
    )

    quaternions = np.stack(
        (
            np.asarray(vertices["rot_0"]),
            np.asarray(vertices["rot_1"]),
            np.asarray(vertices["rot_2"]),
            np.asarray(vertices["rot_3"]),
        ),
        axis=1,
    )

    spherical_harmonics_deg0 = np.stack(
        (
            np.asarray(vertices["f_dc_0"]),
            np.asarray(vertices["f_dc_1"]),
            np.asarray(vertices["f_dc_2"]),
        ),
        axis=1,
    )

    colors = convert_spherical_harmonics_to_rgb(spherical_harmonics_deg0)

    opacity_logits = np.asarray(vertices["opacity"])[..., None]

    supplement_elements = [element for element in plydata.elements if element.name != "vertex"]
    supplement_data: dict[str, Any] = {}
    supplement_keys = ["extrinsic", "intrinsic", "color_space", "image_size"]

    for element in supplement_elements:
        for key in supplement_keys:
            if key not in supplement_data and key in element:
                supplement_data[key] = np.asarray(element[key])

    # Parse intrinsics and image_size.
    if "intrinsic" in supplement_data:
        intrinsics_data = supplement_data["intrinsic"]

        # Legacy: image_size is contained in intrinsic element.
        if "image_size" not in supplement_data:
            if len(intrinsics_data) != 4:
                raise ValueError(
                    "Expect legacy intrinsics with len=4 containing image size, "
                    f"but received len={len(intrinsics_data)}"
                )
            focal_length_px = (intrinsics_data[0], intrinsics_data[1])
            width = int(intrinsics_data[2])
            height = int(intrinsics_data[3])

        else:
            if len(intrinsics_data) != 9:
                raise ValueError(
                    "Expect 9 elements in intrinsics, " f"but received {len(intrinsics_data)}."
                )
            intrinsics_matrix = intrinsics_data.reshape((3, 3))
            focal_length_px = (intrinsics_matrix[0, 0], intrinsics_matrix[1, 1])

            image_size_data = supplement_data["image_size"]
            width = image_size_data[0]
            height = image_size_data[1]

    # Default to VGA resolution: focal length = 512, image size = (640, 480).
    else:
        focal_length_px = (512, 512)
        width = 640
        height = 480

    # Parse extrinsics.
    extrinsics_data = supplement_data.get("extrinsic", np.eye(4).flatten())
    extrinsics_matrix = np.eye(4)

    # Legacy: extrinsics store 12 elements.
    if len(extrinsics_data) == 12:
        extrinsics_matrix[:3] = extrinsics_data.reshape((3, 4))
        extrinsics_matrix[:3, :3] = extrinsics_matrix[:3, :3].copy().T
    elif len(extrinsics_data) == 16:
        extrinsics_matrix[:] = extrinsics_data.reshape((4, 4))
    else:
        raise ValueError(f"Unrecognized extrinsics matrix shape {len(extrinsics_data)}")

    # Parse color space.
    color_space_index = supplement_data.get("color_space", 1)
    color_space = cs_utils.decode_color_space(color_space_index)
    colors = torch.from_numpy(colors).view(1, -1, 3).float()

    if color_space == "sRGB":
        # Convert to linearRGB for proper alpha blending.
        colors = cs_utils.sRGB2linearRGB(colors.flatten(0, 1)).view(1, -1, 3)
        color_space = "linearRGB"

    mean_vectors = torch.from_numpy(mean_vectors).view(1, -1, 3).float()
    quaternions = torch.from_numpy(quaternions).view(1, -1, 4).float()
    singular_values = torch.exp(torch.from_numpy(scale_logits).view(1, -1, 3)).float()
    opacities = torch.sigmoid(torch.from_numpy(opacity_logits).view(1, -1)).float()

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        quaternions=quaternions,
        singular_values=singular_values,
        opacities=opacities,
        colors=colors,
    )
    metadata = SceneMetaData(focal_length_px[0], (width, height), color_space)
    return gaussians, metadata


@torch.no_grad()
def save_ply(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
    path: Path,
    metrics: Metrics | None = None,
) -> PlyData:
    """Save a predicted Gaussian3D to a ply file."""
    start_time = perf_counter() if metrics else None

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)

    # SHARP takes an image, convert it to sRGB color space as input,
    # and predicts linearRGB Gaussians as output.
    # The SHARP renderer would blend linearRGB Gaussians and convert rendered images and videos
    # back to sRGB for the best display quality.
    #
    # However, public renderers do not have such linear2sRGB conversions after rendering.
    # If they render linearRGB Gaussians as-is, the output would be dark without Gamma correction.
    #
    # To make it compatible to public renderers, we force convert linearRGB to sRGB during export.
    # - The SHARP renderer will still handle conversions properly.
    # - Public renderers will be mostly working fine when regarding sRGB images as linearRGB images,
    #   although for the best performance, it is recommended to apply the conversions.
    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    color_space_index = cs_utils.encode_color_space("sRGB")

    # Store opacity logits.
    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)

    attributes = torch.cat(
        (
            xyz,
            colors,
            opacity_logits,
            scale_logits,
            quaternions,
        ),
        dim=1,
    )

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_elements = PlyElement.describe(elements, "vertex")

    # Load image-wise metadata.
    image_height, image_width = image_shape

    # Export image size.
    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    # Export intrinsics.
    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    # Export dummy extrinsics.
    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    # Export number of frames and particles per frame.
    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    # Export disparity ranges for transform.
    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)

    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    quantiles = (
        torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
        .float()
        .cpu()
        .numpy()
    )
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    # Export colorspace.
    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )

    plydata.write(path)
    if metrics and start_time is not None:
        metrics.add_time("save_ply", perf_counter() - start_time)
    return plydata
