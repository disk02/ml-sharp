from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from sharp.models.predictor import RGBGaussianPredictor


class DummyMonodepth(nn.Module):
    def __init__(self, disparity: torch.Tensor) -> None:
        super().__init__()
        self.disparity = disparity

    def forward(self, image: torch.Tensor):
        return SimpleNamespace(
            disparity=self.disparity.to(device=image.device, dtype=image.dtype),
            output_features=torch.zeros_like(image),
            decoder_features=None,
        )


class DummyInitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_depth: torch.Tensor | None = None

    def forward(self, image: torch.Tensor, depth: torch.Tensor):
        self.last_depth = depth.detach().clone()
        return SimpleNamespace(
            feature_input=torch.zeros_like(image),
            gaussian_base_values=SimpleNamespace(),
            global_scale=None,
        )


class DummyFeatureModel(nn.Module):
    def forward(self, feature_input: torch.Tensor, encodings: torch.Tensor | None = None):
        return torch.zeros_like(feature_input)


class DummyPredictionHead(nn.Module):
    def forward(self, image_features: torch.Tensor):
        return image_features


class DummyGaussianComposer(nn.Module):
    def forward(self, delta, base_values, global_scale):
        return SimpleNamespace(
            delta=delta, base_values=base_values, global_scale=global_scale
        )


class DummyScaleMapEstimator(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, monodepth: torch.Tensor, depth: torch.Tensor, decoder_features=None):
        return torch.full_like(monodepth, self.scale)


def test_depth_override_used():
    image = torch.zeros(1, 3, 4, 4)
    disparity = torch.ones(1, 1, 2, 2)
    init_model = DummyInitModel()
    predictor = RGBGaussianPredictor(
        init_model=init_model,
        monodepth_model=DummyMonodepth(disparity),
        feature_model=DummyFeatureModel(),
        prediction_head=DummyPredictionHead(),
        gaussian_composer=DummyGaussianComposer(),
        scale_map_estimator=DummyScaleMapEstimator(scale=2.0),
    )

    depth_override = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, -1.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, float("inf")],
        ]
    ).unsqueeze(0)
    disparity_factor = torch.tensor([2.0])

    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
    )

    expected = depth_override.unsqueeze(1)
    expected = torch.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
    expected = F.interpolate(expected, size=(2, 2), mode="nearest")
    invalid_mask = expected <= 0
    expected = expected.clamp(min=1e-4, max=1e4)
    expected = torch.where(
        invalid_mask,
        torch.tensor(1e-4, device=expected.device, dtype=expected.dtype),
        expected,
    )

    torch.testing.assert_close(init_model.last_depth, expected)


def test_no_override_uses_depth_alignment():
    image = torch.zeros(1, 3, 4, 4)
    disparity = torch.full((1, 1, 2, 2), 2.0)
    init_model = DummyInitModel()
    predictor = RGBGaussianPredictor(
        init_model=init_model,
        monodepth_model=DummyMonodepth(disparity),
        feature_model=DummyFeatureModel(),
        prediction_head=DummyPredictionHead(),
        gaussian_composer=DummyGaussianComposer(),
        scale_map_estimator=DummyScaleMapEstimator(scale=3.0),
    )

    disparity_factor = torch.tensor([2.0])
    depth = torch.ones(1, 1, 2, 2)

    predictor(image=image, disparity_factor=disparity_factor, depth=depth)

    expected_monodepth = disparity_factor[:, None, None, None] / disparity
    expected = expected_monodepth * 3.0
    torch.testing.assert_close(init_model.last_depth, expected)
