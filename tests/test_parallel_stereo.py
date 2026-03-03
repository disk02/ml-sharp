import torch

from sharp.utils.camera import compute_parallel_stereo_pair, create_camera_model
from sharp.utils.gaussians import Gaussians3D


def _make_scene(num_points: int = 256) -> Gaussians3D:
    z = torch.linspace(2.0, 6.0, num_points)
    x = torch.linspace(-0.5, 0.5, num_points)
    y = torch.zeros(num_points)
    means = torch.stack([x, y, z], dim=-1).unsqueeze(0)
    return Gaussians3D(
        mean_vectors=means,
        singular_values=torch.ones(1, num_points, 3),
        quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(1, num_points, 4).clone(),
        colors=torch.zeros(1, num_points, 3),
        opacities=torch.ones(1, num_points, 1),
    )


def test_parallel_stereo_pair_has_shared_rotation_and_off_axis_intrinsics() -> None:
    scene = _make_scene()
    intrinsics = torch.tensor(
        [
            [900.0, 0.0, 511.5, 0.0],
            [0.0, 900.0, 383.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model = create_camera_model(scene, intrinsics, resolution_px=(1024, 768))

    eye_mid = torch.tensor([0.1, -0.03, 0.2], dtype=torch.float32)
    baseline = 0.065
    convergence_depth = 3.25

    left, right = compute_parallel_stereo_pair(
        model,
        eye_mid,
        baseline,
        convergence_depth=convergence_depth,
    )

    rot_l = left.extrinsics[:3, :3]
    rot_r = right.extrinsics[:3, :3]
    assert torch.allclose(rot_l, rot_r, atol=1e-6)

    delta_t = right.extrinsics[:3, 3] - left.extrinsics[:3, 3]
    expected_delta_t = torch.tensor([-baseline, 0.0, 0.0], dtype=delta_t.dtype)
    assert torch.allclose(delta_t, expected_delta_t, atol=1e-5)

    fx = float(intrinsics[0, 0])
    expected_shift = fx * (baseline * 0.5) / convergence_depth
    cx_l = float(left.intrinsics[0, 2])
    cx_r = float(right.intrinsics[0, 2])
    base_cx = float(intrinsics[0, 2])
    assert abs(cx_l - (base_cx + expected_shift)) < 1e-5
    assert abs(cx_r - (base_cx - expected_shift)) < 1e-5


def test_parallel_stereo_pair_zero_baseline_produces_identical_views() -> None:
    scene = _make_scene()
    intrinsics = torch.eye(4, dtype=torch.float32)
    model = create_camera_model(scene, intrinsics, resolution_px=(64, 64))

    eye_mid = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    left, right = compute_parallel_stereo_pair(model, eye_mid, baseline=0.0)

    assert torch.allclose(left.extrinsics, right.extrinsics, atol=1e-6)
    assert torch.allclose(left.intrinsics, right.intrinsics, atol=1e-6)
