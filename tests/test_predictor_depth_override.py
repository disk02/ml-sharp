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
    expected = expected.float()
    expected = torch.where(torch.isfinite(expected), expected, expected.new_tensor(0.0))
    expected = F.interpolate(expected, size=(2, 2), mode="nearest")
    invalid_mask = (~torch.isfinite(expected)) | (expected <= 0)
    expected = torch.where(invalid_mask, expected.new_tensor(1e-4), expected)
    expected = expected.clamp(min=1e-4, max=1e4)

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


def test_disparity_override_conversion():
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

    depth_override = torch.full((1, 4, 4), 2.0)
    disparity_factor = torch.tensor([4.0])
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
        depth_override_is_disparity=True,
    )

    expected = torch.full((1, 1, 2, 2), 2.0)
    torch.testing.assert_close(init_model.last_depth, expected)


def test_monodepth_fallback_replaces_invalid_pixels():
    image = torch.zeros(1, 3, 4, 4)
    disparity = torch.full((1, 1, 2, 2), 2.0)
    init_model = DummyInitModel()
    predictor = RGBGaussianPredictor(
        init_model=init_model,
        monodepth_model=DummyMonodepth(disparity),
        feature_model=DummyFeatureModel(),
        prediction_head=DummyPredictionHead(),
        gaussian_composer=DummyGaussianComposer(),
        scale_map_estimator=DummyScaleMapEstimator(scale=2.0),
    )

    depth_override = torch.tensor([[[1.0, 0.0], [-1.0, 4.0]]])
    disparity_factor = torch.tensor([4.0])
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
        depth_override_fill_mode="monodepth_fallback",
    )

    expected = torch.tensor([[[[1.0, 2.0], [2.0, 4.0]]]])
    torch.testing.assert_close(init_model.last_depth, expected)


def test_depth_override_kept_float32():
    image = torch.zeros(1, 3, 4, 4, dtype=torch.float16)
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

    depth_override = torch.full((1, 4, 4), 2.0, dtype=torch.float16)
    disparity_factor = torch.tensor([4.0], dtype=torch.float16)
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
    )

    assert init_model.last_depth is not None
    assert init_model.last_depth.dtype == torch.float32


def test_depth_override_calibration_per_image_maps_to_monodepth():
    image = torch.zeros(1, 3, 4, 4)
    disparity = torch.full((1, 1, 8, 8), 2.0)
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
        [[[0.1, 0.5, 1.0, 0.25], [0.75, 0.2, 0.9, 0.3]]]
    )
    disparity_factor = torch.tensor([4.0])
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
        depth_override_calibration="per_image",
        depth_override_calibration_percentiles=(0.0, 100.0),
    )

    expected = torch.full((1, 1, 8, 8), 2.0)
    torch.testing.assert_close(init_model.last_depth, expected)


def test_depth_override_calibration_per_sequence_shared_transform():
    image = torch.zeros(2, 3, 4, 4)
    disparity = torch.full((2, 1, 8, 8), 2.0)
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
            [[0.1, 0.5], [1.0, 0.25]],
            [[0.5, 1.0], [1.5, 0.75]],
        ]
    )
    disparity_factor = torch.tensor([4.0, 4.0])
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
        depth_override_calibration="per_sequence",
        depth_override_calibration_percentiles=(0.0, 100.0),
    )

    expected = torch.full((2, 1, 8, 8), 2.0)
    torch.testing.assert_close(init_model.last_depth, expected)


def test_depth_override_calibration_skips_when_insufficient_valid():
    image = torch.zeros(1, 3, 4, 4)
    disparity = torch.full((1, 1, 2, 2), 2.0)
    init_model = DummyInitModel()
    predictor = RGBGaussianPredictor(
        init_model=init_model,
        monodepth_model=DummyMonodepth(disparity),
        feature_model=DummyFeatureModel(),
        prediction_head=DummyPredictionHead(),
        gaussian_composer=DummyGaussianComposer(),
        scale_map_estimator=DummyScaleMapEstimator(scale=2.0),
    )

    depth_override = torch.tensor([[[0.5]]])
    disparity_factor = torch.tensor([4.0])
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
        depth_override_calibration="per_image",
        depth_override_calibration_percentiles=(0.0, 100.0),
    )

    expected = torch.full((1, 1, 2, 2), 0.5)
    torch.testing.assert_close(init_model.last_depth, expected)


def test_depth_override_calibration_multi_channel_disparity():
    image = torch.zeros(1, 3, 4, 4)
    disparity = torch.stack(
        [torch.ones(1, 8, 8), torch.full((1, 8, 8), 2.0)], dim=1
    )
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
        [[[0.1, 0.5, 1.0, 0.25], [0.75, 0.2, 0.9, 0.3]]]
    )
    disparity_factor = torch.tensor([4.0])
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
        depth_override_calibration="per_image",
        depth_override_calibration_percentiles=(0.0, 100.0),
    )

    assert init_model.last_depth is not None
    assert init_model.last_depth.shape == (1, 1, 8, 8)
    assert torch.isfinite(init_model.last_depth).all()


def test_depth_override_fallback_multi_channel_disparity():
    image = torch.zeros(1, 3, 4, 4)
    disparity = torch.stack(
        [torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0)], dim=1
    )
    init_model = DummyInitModel()
    predictor = RGBGaussianPredictor(
        init_model=init_model,
        monodepth_model=DummyMonodepth(disparity),
        feature_model=DummyFeatureModel(),
        prediction_head=DummyPredictionHead(),
        gaussian_composer=DummyGaussianComposer(),
        scale_map_estimator=DummyScaleMapEstimator(scale=2.0),
    )

    depth_override = torch.tensor([[[0.0, 0.0], [1.0, -1.0]]])
    disparity_factor = torch.tensor([4.0])
    predictor(
        image=image,
        disparity_factor=disparity_factor,
        depth_override=depth_override,
        depth_override_fill_mode="monodepth_fallback",
    )

    assert init_model.last_depth is not None
    assert init_model.last_depth.shape == (1, 1, 2, 2)
    assert torch.isfinite(init_model.last_depth).all()
