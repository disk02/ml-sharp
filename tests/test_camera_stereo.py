import torch

from sharp.utils.camera import PinholeCameraModel
from sharp.utils.gaussians import Gaussians3D


def _dummy_scene() -> Gaussians3D:
    mean_vectors = torch.tensor(
        [[0.0, 0.0, 2.0], [0.5, 0.0, 3.0], [-0.5, 0.0, 4.0]], dtype=torch.float32
    )
    count = mean_vectors.shape[0]
    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=torch.ones((count, 3), dtype=torch.float32),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, dtype=torch.float32),
        colors=torch.ones((count, 3), dtype=torch.float32),
        opacities=torch.ones((count,), dtype=torch.float32),
    )


def _camera_model(screen_extrinsics: torch.Tensor | None = None) -> PinholeCameraModel:
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 40.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    if screen_extrinsics is None:
        screen_extrinsics = torch.eye(4, dtype=torch.float32)

    return PinholeCameraModel(
        scene=_dummy_scene(),
        screen_extrinsics=screen_extrinsics,
        screen_intrinsics=intrinsics,
        screen_resolution_px=(100, 80),
    )


def test_parallel_stereo_applies_off_axis_intrinsic_shift() -> None:
    model = _camera_model()
    left, right = model.compute_stereo_pair(
        eye_mid=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        baseline=0.2,
        stereo_mode="parallel",
        stereo_convergence_depth=2.0,
    )

    expected_shift = 100.0 * (0.2 / 2.0) / 2.0
    assert torch.allclose(left.extrinsics[:3, :3], right.extrinsics[:3, :3])
    assert torch.isclose(left.intrinsics[0, 2], torch.tensor(50.0 - expected_shift))
    assert torch.isclose(right.intrinsics[0, 2], torch.tensor(50.0 + expected_shift))


def test_convergence_depth_wins_over_norm() -> None:
    model = _camera_model()
    resolved = model.resolve_convergence_depth(
        stereo_convergence_depth=3.0,
        stereo_convergence_norm=2.0,
    )
    assert resolved == 3.0


def test_parallel_stereo_preserves_screen_extrinsics_composition() -> None:
    angle = torch.tensor(0.3, dtype=torch.float32)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    screen_extrinsics = torch.tensor(
        [
            [cos, -sin, 0.0, 0.2],
            [sin, cos, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model = _camera_model(screen_extrinsics=screen_extrinsics)

    eye_mid = torch.tensor([0.1, 0.0, 0.2], dtype=torch.float32)
    baseline = 0.2
    left, right = model.compute_stereo_pair(
        eye_mid=eye_mid,
        baseline=baseline,
        stereo_mode="parallel",
    )

    midpoint = model.compute(eye_mid)
    expected_rotation = midpoint.extrinsics[:3, :3]

    eye_left = eye_mid.clone()
    eye_left[0] -= baseline * 0.5
    eye_right = eye_mid.clone()
    eye_right[0] += baseline * 0.5

    assert torch.allclose(left.extrinsics[:3, :3], expected_rotation)
    assert torch.allclose(right.extrinsics[:3, :3], expected_rotation)
    assert torch.allclose(left.extrinsics[:3, 3], -(expected_rotation @ eye_left))
    assert torch.allclose(right.extrinsics[:3, 3], -(expected_rotation @ eye_right))
