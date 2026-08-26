"""Unit tests for the close-up frame-edge coverage fix (Defects 1-3).

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import torch

from sharp.models.composer import GaussianComposer
from sharp.models.initializer import GaussianBaseValues, _create_base_xy
from sharp.models.params import (
    DeltaFactor,
    EdgeCoverParams,
)


def _composer(edge_params) -> GaussianComposer:
    return GaussianComposer(
        delta_factor=DeltaFactor(),
        min_scale=0.0,
        max_scale=100.0,
        color_activation_type="linear",
        opacity_activation_type="sigmoid",
        color_space="sRGB",
        base_scale_on_predicted_mean=False,
        scale_factor=1,
        stride=0,
        edge_params=edge_params,
    )


def _base_values(batch: int, layers: int, rows: int, cols: int) -> GaussianBaseValues:
    shape = (batch, 1, layers, rows, cols)
    ones = torch.ones(*shape)
    # Quaternions live in the channel dim: 4 components, shared across layers.
    quats = torch.zeros(batch, 4, 1, rows, cols)
    quats[:, 0] = 1.0
    # Colors: 3 channels in the channel dim, per layer.
    colors = torch.zeros(batch, 3, layers, rows, cols)
    colors[...] = 0.5
    return GaussianBaseValues(
        mean_x_ndc=torch.zeros(*shape),
        mean_y_ndc=torch.zeros(*shape),
        mean_inverse_z_ndc=ones,
        scales=ones,
        quaternions=quats,
        colors=colors,
        opacities=ones * 0.5,
    )


def _border_mask(rows: int, cols: int, n: int = 1) -> torch.Tensor:
    border = torch.zeros(rows, cols, dtype=torch.bool)
    border[:n, :] = border[-n:, :] = border[:, :n] = border[:, -n:] = True
    return border


def test_edge_ring_snap_left_to_frame_boundary() -> None:
    stride, height, width, num_layers = 8, 96, 128, 2
    depth = torch.rand(1, 1, height, width)

    x_ring, y_ring = _create_base_xy(depth, stride, num_layers, edge_ring=True)
    x_plain, y_plain = _create_base_xy(depth, stride, num_layers, edge_ring=False)

    # Uns-ringed grid begins half a stride inside the frame; snapping it out
    # is Defect 1. Both left and top extremes must now be exactly NDC -1.0.
    assert torch.isclose(x_ring.amin(), torch.tensor(-1.0))
    assert torch.isclose(y_ring.amin(), torch.tensor(-1.0))
    # And plain (inside) extends strictly less far.
    assert torch.isclose(x_plain.amin(), torch.tensor(-1.0 + 2 * (0.5 * stride) / width))
    assert x_ring.amin() < x_plain.amin()
    assert y_ring.amin() < y_plain.amin()


def test_edge_ring_snap_right_and_top_to_last_pixel() -> None:
    stride, height, width, num_layers = 8, 96, 128, 2
    depth = torch.rand(1, 1, height, width)

    x_ring, y_ring = _create_base_xy(depth, stride, num_layers, edge_ring=True)
    x_plain, y_plain = _create_base_xy(depth, stride, num_layers, edge_ring=False)

    # Right/bottom extremes must advance from the last inner stride point to
    # the last pixel centre, i.e. NDC ≈ 1 - stride/W (not > 1). They must
    # exceed plain but not overlap past the frame.
    assert x_ring.amax() > x_plain.amax()
    assert y_ring.amax() > y_plain.amax()
    assert torch.isclose(
        x_ring.amax(), torch.tensor(1.0 - 2.0 / width)
    )
    assert torch.isclose(
        y_ring.amax(), torch.tensor(1.0 - 2.0 / height)
    )


def test_edge_ring_preserves_shape() -> None:
    stride, height, width, num_layers = 8, 96, 128, 2
    depth = torch.rand(1, 1, height, width)

    x_ring, y_ring = _create_base_xy(depth, stride, num_layers, edge_ring=True)
    assert x_ring.shape == y_ring.shape
    assert x_ring.shape[-2:] == (height // stride, width // stride)


def test_edge_veil_scales_and_dims_border() -> None:
    num_layers, rows, cols = 2, 6, 8
    e = EdgeCoverParams(px=1, veil_scale=1.5, veil_alpha=0.5)
    bv = _base_values(1, num_layers, rows, cols)
    delta = torch.zeros(1, 14, num_layers, rows, cols)

    plain = _composer(None)(delta, bv, flatten_output=False)
    veiled = _composer(e)(delta, bv, flatten_output=False)

    border = _border_mask(rows, cols)
    inner = ~border

    scale_ratio = veiled.singular_values / plain.singular_values
    alpha_ratio = veiled.opacities / plain.opacities

    # The veil pads scale 1.5x and dims alpha 0.5x on border cells only.
    assert torch.allclose(scale_ratio[..., border], torch.ones_like(scale_ratio[..., border]) * 1.5)
    assert torch.allclose(alpha_ratio[..., border], torch.ones_like(alpha_ratio[..., border]) * 0.5)

    # Interior must be numerically identical (mask == 0 there). With the
    # linear/sigmoid activations used here, the only possible interior diff
    # is the sign of a 0.0 — compare by absolute value for that.
    interior_diff = (scale_ratio - 1.0).abs()
    assert (interior_diff[..., inner] < 1e-7).all()
    opacity_diff = (alpha_ratio - 1.0).abs()
    assert (opacity_diff[..., inner] < 1e-7).all()

    # Regression guard for the prune stage: the opacities flat count must match
    # the singular_values flat count (else (opacities>=min_op) & (max_scale>=min)
    # splits). The old 5-D veil mask broadcast opacities to an extra batch dim.
    on_f = _composer(e)(torch.zeros(1, 14, num_layers, rows, cols), bv)  # flatten_output=True
    off_f = _composer(None)(torch.zeros(1, 14, num_layers, rows, cols), bv)
    op_cells = on_f.opacities.shape[-2] * on_f.opacities.shape[-1] * on_f.opacities.shape[0]
    sv_cells = on_f.singular_values.shape[:-1].numel()
    assert op_cells == sv_cells, f"opacities {op_cells} != singular {sv_cells}"
    # And identical when off.
    _fin = off_f.opacities.shape[-2] * off_f.opacities.shape[-1] * off_f.opacities.shape[0]
    assert _fin == on_f.singular_values.shape[:-1].numel()


def test_edge_veil_disabled_is_noop() -> None:
    num_layers, rows, cols = 2, 6, 8
    e = EdgeCoverParams(enabled=False, px=2, veil_scale=1.5, veil_alpha=0.5)
    bv = _base_values(1, num_layers, rows, cols)
    delta = torch.zeros(1, 14, num_layers, rows, cols)

    off = _composer(e)(delta, bv, flatten_output=False)
    empty = _composer(None)(delta, bv, flatten_output=False)

    assert torch.equal(off.mean_vectors, empty.mean_vectors)
    assert torch.equal(off.singular_values, empty.singular_values)
    assert torch.equal(off.opacities, empty.opacities)
    # Sanity check: the enabled variant actually changes border.
    border = _border_mask(rows, cols)
    on = _composer(EdgeCoverParams(px=2, veil_scale=1.5, veil_alpha=0.5))(delta, bv, flatten_output=False)
    assert not torch.equal(on.singular_values[..., border], empty.singular_values[..., border])