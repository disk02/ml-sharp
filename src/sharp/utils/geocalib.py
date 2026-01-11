"""GeoCalib adapter for focal length estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


def _get_attr(obj: object, name: str) -> Any | None:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_f_px(calibration: object) -> float:
    camera = _get_attr(calibration, "camera")
    if camera is None and isinstance(calibration, dict):
        camera = calibration.get("camera")
    if camera is None:
        camera = calibration
    fx = _get_attr(camera, "fx")
    fy = _get_attr(camera, "fy")
    if fx is not None and fy is not None:
        return float((fx + fy) / 2.0)
    f = _get_attr(camera, "f")
    if f is None and isinstance(calibration, dict):
        f = calibration.get("f")
    if f is not None:
        return float(f)
    raise RuntimeError("GeoCalib calibration did not include focal length values.")


@dataclass
class GeoCalibRunner:
    device: torch.device
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from geocalib import GeoCalib

        self._model = GeoCalib()
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)

    def calibrate_image(self, image_path: Path) -> float:
        image = self._model.load_image(str(image_path))
        if hasattr(image, "to"):
            image = image.to(self.device)
        calibration = self._model.calibrate(image)
        return _extract_f_px(calibration)

    def calibrate_folder(self, image_paths: list[Path]) -> float:
        images = []
        for path in image_paths:
            image = self._model.load_image(str(path))
            if hasattr(image, "to"):
                image = image.to(self.device)
            images.append(image)
        calibration = self._model.calibrate(images, shared_intrinsics=True)
        return _extract_f_px(calibration)
