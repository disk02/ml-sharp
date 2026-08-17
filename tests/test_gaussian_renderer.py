import numpy as np
import pytest
import torch
from sharp.rendering import gaussian_renderer
from sharp.utils.metrics import RenderTiming


def test_pinhole_intrinsics_values() -> None:
    device = torch.device("cpu")
    intrinsic = gaussian_renderer._pinhole_intrinsics(123.5, 640, 480, device)
    assert intrinsic.shape == (4, 4)
    assert intrinsic.dtype == torch.float32
    assert intrinsic.device == device
    expected = torch.tensor(
        [
            [123.5, 0.0, 319.5, 0.0],
            [0.0, 123.5, 239.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(intrinsic, expected)


def test_finalize_sbs_frame_packs_left_right() -> None:
    color_l = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]], dtype=torch.uint8
    )
    color_r = torch.tensor(
        [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]], dtype=torch.uint8
    )
    sbs = gaussian_renderer._finalize_sbs_frame(
        color_l, color_r, align_crop=False, render_timing=None
    )
    assert sbs.shape == (2, 4, 3)
    assert sbs.dtype == np.uint8
    np.testing.assert_array_equal(sbs[:, :2, :], color_l.numpy())
    np.testing.assert_array_equal(sbs[:, 2:, :], color_r.numpy())


def test_finalize_sbs_frame_records_timing_stages() -> None:
    render_timing = RenderTiming()
    color_l = torch.zeros((4, 4, 3), dtype=torch.uint8)
    color_r = torch.zeros((4, 4, 3), dtype=torch.uint8)
    render_timing.start_frame()
    gaussian_renderer._finalize_sbs_frame(
        color_l, color_r, align_crop=False, render_timing=render_timing
    )
    render_timing.finalize_frame()
    assert "render_d2h_transfer" in render_timing.timings
    assert "render_encode_prepare" in render_timing.timings


def _disk_frame(offset: int) -> torch.Tensor:
    yy, xx = np.mgrid[0:64, 0:64]
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = (xx - 32 - offset) ** 2 + (yy - 32) ** 2 <= 100
    frame[mask, :] = 255
    return torch.from_numpy(frame)


def test_finalize_sbs_frame_align_crop() -> None:
    pytest.importorskip("cv2")
    sbs = gaussian_renderer._finalize_sbs_frame(
        _disk_frame(0), _disk_frame(2), align_crop=True, render_timing=None
    )
    assert sbs.ndim == 3
    assert sbs.dtype == np.uint8
    assert sbs.shape[0] >= 8
    assert sbs.shape[1] % 2 == 0
    assert 8 <= sbs.shape[1] <= 128
    assert sbs.shape[2] == 3


def test_timed_helpers_noop_without_metrics() -> None:
    color_l = torch.zeros((4, 4, 3), dtype=torch.uint8)
    color_r = torch.zeros((4, 4, 3), dtype=torch.uint8)
    with gaussian_renderer._timed_cpu(None, "render_d2h_transfer"):
        with gaussian_renderer._timed_gpu(None, "render_gpu_raster_blend"):
            sbs = gaussian_renderer._finalize_sbs_frame(
                color_l, color_r, align_crop=False, render_timing=None
            )
    assert sbs.shape == (4, 8, 3)
