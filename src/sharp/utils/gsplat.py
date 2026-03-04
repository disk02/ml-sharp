"""Contains utility code for gsplat renderer.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import gsplat
import torch
from torch import nn

from sharp.utils import color_space as cs_utils
from sharp.utils import io, vis
from sharp.utils.gaussians import BackgroundColor, Gaussians3D
from sharp.utils.metrics import RenderTiming


class RenderingOutputs(NamedTuple):
    """Outputs of 3D Gaussians renderer."""

    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


def write_renderings(rendering: RenderingOutputs, output_folder: Path, filename: str):
    """Write rendered color/depth/alpha to files."""
    batch_size = len(rendering.color)
    if batch_size != 1:
        raise RuntimeError("We only support saving rendering of batch size = 1")

    def _save_image_tensor(tensor: torch.Tensor, suffix: str):
        np_array = tensor.permute(1, 2, 0).numpy()
        io.save_image(np_array, (output_folder / filename).with_suffix(suffix))

    color = (rendering.color[0] * 255.0).to(dtype=torch.uint8).cpu()
    colorized_depth = vis.colorize_depth(rendering.depth[0], val_max=100.0)
    colorized_alpha = vis.colorize_alpha(rendering.alpha[0])

    _save_image_tensor(color, ".color.png")
    _save_image_tensor(colorized_depth, ".depth.png")
    _save_image_tensor(colorized_alpha, ".alpha.png")


class GSplatRenderer(nn.Module):
    """Module to render 3D Gaussians to images using gsplat."""

    color_space: cs_utils.ColorSpace
    background_color: BackgroundColor

    def __init__(
        self,
        color_space: cs_utils.ColorSpace = "sRGB",
        background_color: BackgroundColor = "black",
        low_pass_filter_eps: float = 0.0,
    ) -> None:
        """Initialize gsplat renderer.

        Args:
            color_space: The color space to use for rendering.
            background_color: The background color to use for rendering.
            low_pass_filter_eps: The epsilon value for the low pass filter.
        """
        super().__init__()
        self.color_space = color_space
        self.background_color = background_color
        self.low_pass_filter_eps = low_pass_filter_eps

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
        want_depth: bool = True,
        render_timing: RenderTiming | None = None,
    ) -> RenderingOutputs:
        """Predict images from gaussians.

        Args:
            gaussians: The Gaussians to render.
            extrinsics: The extrinsics of the camera to render to in OpenCV format.
            intrinsics: The intriniscs of the camera to render to in OpenCV format.
            image_width: The desired output image width.
            image_height: The desired output image height.
            want_depth: Whether to render depth alongside RGB.
        """
        batch_size = len(gaussians.mean_vectors)
        outputs_list: list[RenderingOutputs] = []

        for ib in range(batch_size):
            render_mode = "RGB+D" if want_depth else "RGB"
            if render_timing is None:
                colors, alphas, meta = gsplat.rendering.rasterization(
                    means=gaussians.mean_vectors[ib],
                    quats=gaussians.quaternions[ib],
                    scales=gaussians.singular_values[ib],
                    opacities=gaussians.opacities[ib],
                    colors=gaussians.colors[ib],
                    viewmats=extrinsics[ib : ib + 1],
                    Ks=intrinsics[ib : ib + 1, :3, :3],
                    width=image_width,
                    height=image_height,
                    render_mode=render_mode,
                    rasterize_mode="classic",
                    absgrad=False,
                    packed=False,
                    eps2d=self.low_pass_filter_eps,
                )
            else:
                # GPU project/sort is dominated by the rasterization kernel.
                with render_timing.gpu_event_timer("render_gpu_project_sort"):
                    colors, alphas, meta = gsplat.rendering.rasterization(
                        means=gaussians.mean_vectors[ib],
                        quats=gaussians.quaternions[ib],
                        scales=gaussians.singular_values[ib],
                        opacities=gaussians.opacities[ib],
                        colors=gaussians.colors[ib],
                        viewmats=extrinsics[ib : ib + 1],
                        Ks=intrinsics[ib : ib + 1, :3, :3],
                        width=image_width,
                        height=image_height,
                        render_mode=render_mode,
                        rasterize_mode="classic",
                        absgrad=False,
                        packed=False,
                        eps2d=self.low_pass_filter_eps,
                    )

            rendered_color = colors[..., 0:3].permute([0, 3, 1, 2])
            rendered_alpha = alphas.permute([0, 3, 1, 2])

            # GPU shading covers background composition and colorspace conversion.
            if render_timing is None:
                rendered_color = self.compose_with_background(
                    rendered_color, rendered_alpha, self.background_color
                )
                if self.color_space == "sRGB":
                    pass
                elif self.color_space == "linearRGB":
                    rendered_color = cs_utils.linearRGB2sRGB(rendered_color)
                else:
                    ValueError("Unsupported ColorSpace type.")
            else:
                with render_timing.gpu_event_timer("render_gpu_shading"):
                    rendered_color = self.compose_with_background(
                        rendered_color, rendered_alpha, self.background_color
                    )
                    if self.color_space == "sRGB":
                        pass
                    elif self.color_space == "linearRGB":
                        rendered_color = cs_utils.linearRGB2sRGB(rendered_color)
                    else:
                        ValueError("Unsupported ColorSpace type.")

            # GPU raster/blend stage includes post-raster depth normalization work.
            if render_timing is None:
                if want_depth:
                    rendered_depth_unnormalized = colors[..., 3:4].permute([0, 3, 1, 2])
                    cov2d = self._conics_to_covars2d(meta["conics"])
                    splats_visible_mask = meta["depths"] > 1e-2
                    cov2d[~splats_visible_mask][..., 0, 0] = 1
                    cov2d[~splats_visible_mask][..., 1, 1] = 1
                    cov2d[~splats_visible_mask][..., 0, 1] = 0
                    rendered_depth = rendered_depth_unnormalized / torch.clip(
                        rendered_alpha, min=1e-8
                    )
                else:
                    rendered_depth = rendered_color.new_empty(
                        rendered_color.shape[0],
                        1,
                        rendered_color.shape[2],
                        rendered_color.shape[3],
                    )
            else:
                with render_timing.gpu_event_timer("render_gpu_raster_blend"):
                    if want_depth:
                        rendered_depth_unnormalized = colors[..., 3:4].permute([0, 3, 1, 2])
                        cov2d = self._conics_to_covars2d(meta["conics"])
                        splats_visible_mask = meta["depths"] > 1e-2
                        cov2d[~splats_visible_mask][..., 0, 0] = 1
                        cov2d[~splats_visible_mask][..., 1, 1] = 1
                        cov2d[~splats_visible_mask][..., 0, 1] = 0
                        rendered_depth = rendered_depth_unnormalized / torch.clip(
                            rendered_alpha, min=1e-8
                        )
                    else:
                        rendered_depth = rendered_color.new_empty(
                            rendered_color.shape[0],
                            1,
                            rendered_color.shape[2],
                            rendered_color.shape[3],
                        )

            outputs = RenderingOutputs(
                color=rendered_color,
                depth=rendered_depth,
                alpha=rendered_alpha,
            )
            outputs_list.append(outputs)

        return RenderingOutputs(
            color=torch.cat([item.color for item in outputs_list], dim=0).contiguous(),
            depth=torch.cat([item.depth for item in outputs_list], dim=0).contiguous(),
            alpha=torch.cat([item.alpha for item in outputs_list], dim=0).contiguous(),
        )


    def render_with_rays(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
        rays_cam: torch.Tensor,
        want_depth: bool = True,
        render_timing: RenderTiming | None = None,
    ) -> RenderingOutputs:
        """Render by warping pinhole output to custom camera-space rays.

        This is used as a practical fallback for cylindrical projection where
        gsplat does not expose per-pixel ray inputs.
        """
        base = self.forward(
            gaussians=gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_width=image_width,
            image_height=image_height,
            want_depth=want_depth,
            render_timing=render_timing,
        )

        if rays_cam.shape != (image_height, image_width, 3):
            raise ValueError(
                f"rays_cam must have shape {(image_height, image_width, 3)}, got {tuple(rays_cam.shape)}"
            )

        z = rays_cam[..., 2].clamp(min=1e-8)
        x_img = rays_cam[..., 0] / z
        y_img = rays_cam[..., 1] / z

        fx = intrinsics[0, 0, 0]
        fy = intrinsics[0, 1, 1]
        cx = intrinsics[0, 0, 2]
        cy = intrinsics[0, 1, 2]

        u = fx * x_img + cx
        v = fy * y_img + cy

        x_norm = (u / max(image_width - 1, 1)) * 2.0 - 1.0
        y_norm = (v / max(image_height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

        warped_color = torch.nn.functional.grid_sample(
            base.color,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_alpha = torch.nn.functional.grid_sample(
            base.alpha,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_depth = torch.nn.functional.grid_sample(
            base.depth,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return RenderingOutputs(color=warped_color, depth=warped_depth, alpha=warped_alpha)

    @staticmethod
    def compose_with_background(
        rendered_rgb: torch.Tensor,
        rendered_alpha: torch.Tensor,
        background_color: BackgroundColor,
    ) -> torch.Tensor:
        """Compose rendered RGB with background color."""
        if background_color == "black":
            return rendered_rgb
        elif background_color == "white":
            return rendered_rgb + (1.0 - rendered_alpha)
        elif background_color == "random_color":
            return (
                rendered_rgb
                + (1.0 - rendered_alpha)
                * torch.rand(3, dtype=rendered_rgb.dtype, device=rendered_rgb.device)[
                    None, :, None, None
                ]
            )
        elif background_color == "random_pixel":
            return rendered_rgb + (1.0 - rendered_alpha) * torch.rand_like(rendered_rgb)
        else:
            raise ValueError("Unsupported BackgroundColor type.")

    @staticmethod
    def _conics_to_covars2d(conics: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """Convert conics to covariance matrices."""
        a = conics[..., 0]
        b = conics[..., 1]
        c = conics[..., 2]
        # Reconstruct determinant.
        det = 1 / (a * c - b**2 + eps)
        det = det.clamp(min=eps)
        # Reconstruct covars2d.
        covars2d = torch.zeros(*conics.shape[:-1], 2, 2, device=conics.device)
        covars2d[..., 1, 1] = a * det
        covars2d[..., 0, 0] = c * det
        covars2d[..., 0, 1] = -b * det
        covars2d[..., 1, 0] = -b * det
        covars2d = torch.nan_to_num(covars2d, nan=0.0, posinf=0.0, neginf=0.0)
        return covars2d
