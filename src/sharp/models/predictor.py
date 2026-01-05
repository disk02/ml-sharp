"""Contains definition of RGB-only gaussian predictor.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from sharp.models.monodepth import MonodepthWithEncodingAdaptor
from sharp.utils.gaussians import Gaussians3D

from .composer import GaussianComposer

LOGGER = logging.getLogger(__name__)


class DepthAlignment(nn.Module):
    """Depth alignment in a dedicated nn.Module.

    Wrap scale_map_estimator to perform the conditional logic in a separated torch
    module outside the forward of RGBGaussianPredictor. This module can be then
    excluded during symbolic tracing.
    """

    def __init__(self, scale_map_estimator: nn.Module | None):
        """Initialize DepthAlignmentWrapper.

        Args:
            scale_map_estimator: Module to align monodepth to ground truth depth.
        """
        super().__init__()
        self.scale_map_estimator = scale_map_estimator

    def forward(
        self,
        monodepth: torch.Tensor,
        depth: torch.Tensor,
        depth_decoder_features: torch.Tensor | None = None,
    ):
        """Optionally align monodepth to ground truth with a local scale map.

        Args:
            monodepth: The monodepth model with intermediate features to use.
            depth: Ground truth depth to align predicted depth to.
            depth_decoder_features: The (optional) monodepth decoder features.
        """
        if depth is not None and self.scale_map_estimator is not None:
            depth_alignment_map = self.scale_map_estimator(
                monodepth[:, 0:1], depth, depth_decoder_features
            )
            monodepth = depth_alignment_map * monodepth
        else:
            # Some losses rely on the presence of an alignment map.
            # We ensure that they can be computed by creating a fake alignment map.
            depth_alignment_map = torch.ones_like(monodepth)
        return monodepth, depth_alignment_map


class RGBGaussianPredictor(nn.Module):
    """Predicts 3D Gaussians from images."""

    feature_model: nn.Module

    def __init__(
        self,
        init_model: nn.Module,
        monodepth_model: MonodepthWithEncodingAdaptor,
        feature_model: nn.Module,
        prediction_head: nn.Module,
        gaussian_composer: GaussianComposer,
        scale_map_estimator: nn.Module | None,
    ) -> None:
        """Initialize RGBGaussianPredictor.

        Args:
            init_model: A model mapping image and depth to base values.
            monodepth_model: The monodepth model with intermediate features to use.
            feature_model: The image2image model to predict Gaussians from.
            prediction_head: Head to decode image features.
            gaussian_composer: Module to compose final prediction from deltas and
                base values.
            scale_map_estimator: Module to align monodepth to ground truth depth.

        Note:
        ----
            when monodepth_model is trainable, using local depth alignment can
            result in the monodepth model losing its ability to predict shapes. It is
            hence recommend to deactivate the corresponding flag.
        """
        super().__init__()
        self.init_model = init_model
        self.feature_model = feature_model
        self.monodepth_model = monodepth_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer
        self.depth_alignment = DepthAlignment(scale_map_estimator)

    def forward(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
        depth: torch.Tensor | None = None,
        depth_override: torch.Tensor | None = None,
        depth_override_is_disparity: bool = False,
        depth_override_fill_mode: Literal["override_only", "monodepth_fallback"] = (
            "override_only"
        ),
        depth_override_calibration: Literal["none", "per_image", "per_sequence"] = "none",
        depth_override_calibration_percentiles: tuple[float, float] = (10.0, 90.0),
        depth_override_calibration_min_valid: float = 1e-6,
    ) -> Gaussians3D:
        """Predict 3D Gaussians.

        Args:
            image: The image to process.
            disparity_factor: Factor to convert depth to disparities.
            depth: Ground truth depth to align predicted depth to.
            depth_override: External depth map to use instead of predicted monodepth.
            depth_override_is_disparity: Whether depth_override contains disparity values.
            depth_override_fill_mode: How to handle invalid override values.
            depth_override_calibration: Optional calibration mode for normalized depth inputs.
            depth_override_calibration_percentiles: Percentiles for robust calibration.
            depth_override_calibration_min_valid: Minimum override value considered valid.

        Returns:
            The predicted 3D Gaussians.

        Note:
        ----
        During training, it is recommended to feed an additional ground truth depth
        map to the network to align the predicted depth to. During inference, you may
        either omit depth and use monodepth disparity as before, or provide
        depth_override (metric depth or disparity with depth_override_is_disparity=True)
        to drive geometry initialization.
        """
        # Estimate depth and align to ground truth (if available).
        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity

        disparity_factor_base = disparity_factor
        disparity_factor = disparity_factor_base[:, None, None, None]
        monodepth = disparity_factor / monodepth_disparity.clamp(min=1e-4, max=1e4)

        # In the model we apply additional alignment to provided ground truth depth
        # as well as additional normalization.
        #
        # The overall graph looks as follows:
        #
        #     monodepth        depth    # Both monodepth and depth are metric here.
        #         |              |
        #         +------+-------+
        #                |
        #        +-------+--------+     # Optionally align monodepth to ground truth
        #        |depth_alignement|     # with a local scale map.
        #        +-------+--------+
        #                |
        #                v
        #       monodepth (aligned)     # Monodepth is now aligned to ground truth.
        #                |
        #          +-----+----+         # Normalize depth and compute base gaussians.
        #          |init_model|         # in these normalized coordinates.
        #          +-----+----+
        #                |
        #                v
        #   +------ init_output         # Init_output consists of features, base
        #   |            |              # gaussians and a global scale.
        #   |     +------+-----+
        #   |     |main network|        # Compute delta values to base gaussians.
        #   |     +------+-----+
        #   |            |
        #   |            V
        #   |        delta_values       # The delta values are computed with normalized depth.
        #   |            |
        #   |    +-------+---------+
        #   +--> |gaussian_composer|    # Add delta to base values and unscale gaussians.
        #        +-------+---------+
        #                |
        #                v
        #            gaussians          # The final Gaussians are metric again.
        #

        # The logic to decide whether to align monodepth to the ground truth is wrapped
        # in a submodule 'DepthAlignement' to facilitate the symbolic tracing of the
        # predictor. This way, the depth alignment submodule containing the conditional
        # logic can be excluded during the tracing and the graph of the predictors is
        # static.
        if depth_override is not None:
            depth_used = depth_override
            if depth_used.dim() == 3:
                depth_used = depth_used.unsqueeze(1)
            if depth_used.dim() != 4 or depth_used.shape[1] != 1:
                raise ValueError(
                    "depth_override must have shape [B, 1, H, W] or [B, H, W]."
                )
            if depth_used.shape[0] != image.shape[0]:
                raise ValueError(
                    "depth_override batch size must match image batch size: "
                    f"{depth_used.shape[0]} != {image.shape[0]}"
                )

            depth_used = depth_used.to(device=image.device)
            if depth_used.dtype != torch.float32:
                depth_used = depth_used.float()
            zero = depth_used.new_tensor(0.0)
            depth_used = torch.where(torch.isfinite(depth_used), depth_used, zero)
            if depth_used.shape[-2:] != monodepth_disparity.shape[-2:]:
                depth_used = F.interpolate(
                    depth_used,
                    size=monodepth_disparity.shape[-2:],
                    mode="nearest",
                )

            disparity_factor_f = disparity_factor_base.float()[:, None, None, None]
            if depth_override_is_disparity:
                depth_used = disparity_factor_f / depth_used.clamp(min=1e-4, max=1e4)

            if depth_override_calibration != "none":
                monodepth_f = disparity_factor_f / monodepth_disparity.float().clamp(
                    min=1e-4, max=1e4
                )
                p_lo, p_hi = depth_override_calibration_percentiles
                if depth_override_calibration == "per_sequence":
                    override_valid = torch.isfinite(depth_used) & (
                        depth_used > depth_override_calibration_min_valid
                    )
                    ref_valid = torch.isfinite(monodepth_f) & (monodepth_f > 0)
                    joint_valid = override_valid & ref_valid
                    override_vals = depth_used[joint_valid]
                    ref_vals = monodepth_f[joint_valid]
                    if override_vals.numel() >= 32 and ref_vals.numel() >= 32:
                        o_lo = torch.quantile(override_vals, p_lo / 100.0)
                        o_hi = torch.quantile(override_vals, p_hi / 100.0)
                        r_lo = torch.quantile(ref_vals, p_lo / 100.0)
                        r_hi = torch.quantile(ref_vals, p_hi / 100.0)
                        denom = torch.clamp(o_hi - o_lo, min=1e-6)
                        a = (r_hi - r_lo) / denom
                        b = r_lo - a * o_lo
                        depth_used = torch.where(override_valid, a * depth_used + b, depth_used)
                else:
                    for batch_idx in range(depth_used.shape[0]):
                        override_valid = torch.isfinite(depth_used[batch_idx]) & (
                            depth_used[batch_idx] > depth_override_calibration_min_valid
                        )
                        ref_valid = torch.isfinite(monodepth_f[batch_idx]) & (
                            monodepth_f[batch_idx] > 0
                        )
                        joint_valid = override_valid & ref_valid
                        override_vals = depth_used[batch_idx][joint_valid]
                        ref_vals = monodepth_f[batch_idx][joint_valid]
                        if override_vals.numel() < 32 or ref_vals.numel() < 32:
                            continue
                        o_lo = torch.quantile(override_vals, p_lo / 100.0)
                        o_hi = torch.quantile(override_vals, p_hi / 100.0)
                        r_lo = torch.quantile(ref_vals, p_lo / 100.0)
                        r_hi = torch.quantile(ref_vals, p_hi / 100.0)
                        denom = torch.clamp(o_hi - o_lo, min=1e-6)
                        a = (r_hi - r_lo) / denom
                        b = r_lo - a * o_lo
                        depth_used[batch_idx] = torch.where(
                            override_valid, a * depth_used[batch_idx] + b, depth_used[batch_idx]
                        )

            invalid_mask = (~torch.isfinite(depth_used)) | (depth_used <= 0)
            if depth_override_fill_mode == "monodepth_fallback":
                if depth_override_calibration == "none":
                    monodepth_f = disparity_factor_f / monodepth_disparity.float().clamp(
                        min=1e-4, max=1e4
                    )
                depth_used = torch.where(invalid_mask, monodepth_f, depth_used)
            else:
                eps = depth_used.new_tensor(1e-4)
                depth_used = torch.where(invalid_mask, eps, depth_used)

            depth_used = depth_used.clamp(min=1e-4, max=1e4)
            init_output = self.init_model(image, depth_used)
        else:
            monodepth, _ = self.depth_alignment(
                monodepth,
                depth,
                monodepth_output.decoder_features,
            )

            init_output = self.init_model(image, monodepth)
        image_features = self.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )
        delta_values = self.prediction_head(image_features)
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )
        return gaussians

    def internal_resolution(self) -> int:
        """Internal resolution."""
        return self.monodepth_model.internal_resolution()

    @property
    def output_resolution(self) -> int:
        """Output resolution of Gaussians."""
        return self.internal_resolution() // 2
