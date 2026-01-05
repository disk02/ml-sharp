from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pytest

from sharp.utils.depth_io import load_depth, resolve_depth_for_image


def test_load_depth_npy_applies_scale(tmp_path: Path) -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = tmp_path / "depth.npy"
    np.save(path, data)

    loaded = load_depth(path, scale=0.5)
    np.testing.assert_allclose(loaded, data * 0.5)


def test_load_depth_png_uint16(tmp_path: Path) -> None:
    data = np.array([[1000, 2000], [3000, 0]], dtype=np.uint16)
    path = tmp_path / "depth.png"
    iio.imwrite(path, data)

    loaded = load_depth(path, scale=0.001)
    expected = data.astype(np.float32) * 0.001
    np.testing.assert_allclose(loaded, expected)


def test_resolve_depth_for_image_prefers_npy(tmp_path: Path) -> None:
    image_path = tmp_path / "img_0001.jpg"
    image_path.write_text("fake")
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    (depth_dir / "img_0001.PNG").write_text("fake")
    npy_path = depth_dir / "img_0001.npy"
    np.save(npy_path, np.zeros((2, 2), dtype=np.float32))

    resolved = resolve_depth_for_image(image_path, depth_dir)
    assert resolved == npy_path


def test_resolve_depth_for_image_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "img_0002.jpg"
    image_path.write_text("fake")
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_depth_for_image(image_path, depth_dir)
    message = str(excinfo.value)
    assert "img_0002.jpg" in message
    assert str(depth_dir) in message
