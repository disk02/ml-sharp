import torch

from sharp.utils.camera import (
    compute_parallel_stereo_pair,
    create_camera_model,
    make_cylindrical_rays,
)
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


def test_make_cylindrical_rays_horizontal_mapping() -> None:
    width, height = 101, 51
    hfov_deg = 100.0
    fy = 80.0
    cy = (height - 1) / 2.0
    rays = make_cylindrical_rays(
        width=width,
        height=height,
        hfov_deg=hfov_deg,
        fy=fy,
        cy=cy,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    center = rays[height // 2, width // 2]
    assert abs(float(center[0])) < 1e-4
    expected_center_y = ((height // 2 + 0.5) - cy) / fy
    assert abs(float(center[1] / center[2]) - expected_center_y) < 1e-4
    assert abs(float(center[2]) - 1.0) < 1e-4

    hfov_rad = hfov_deg * torch.pi / 180.0
    expected = torch.tan(torch.tensor(float(hfov_rad / 2.0 - hfov_rad / (2.0 * width))))
    left_x = rays[height // 2, 0, 0] / rays[height // 2, 0, 2]
    right_x = rays[height // 2, -1, 0] / rays[height // 2, -1, 2]
    assert torch.allclose(left_x, -expected, atol=1e-4)
    assert torch.allclose(right_x, expected, atol=1e-4)


def test_cylindrical_base_intrinsics_cover_requested_hfov() -> None:
    width, height = 640, 360
    hfov_deg = 100.0
    cx = (width - 1) / 2.0
    fy = 300.0

    rays = make_cylindrical_rays(
        width=width,
        height=height,
        hfov_deg=hfov_deg,
        fy=fy,
        cy=(height - 1) / 2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    fx_base = (width * 0.5) / torch.tan(torch.tensor(hfov_deg * torch.pi / 180.0 * 0.5))

    rays_center_row = rays[height // 2]
    u_in = fx_base * (rays_center_row[:, 0] / rays_center_row[:, 2]) + cx

    eps = 1e-3
    assert float(u_in.min()) >= -eps
    assert float(u_in.max()) <= (width - 1) + eps


def test_parallel_stereo_pair_intrinsics_override_controls_cx_shift() -> None:
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

    baseline = 0.065
    convergence_depth = 3.25
    fx_base = 420.0
    intrinsics_override = model.screen_intrinsics.clone()
    intrinsics_override[0, 0] = fx_base

    left, right = compute_parallel_stereo_pair(
        model,
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        baseline,
        convergence_depth=convergence_depth,
        intrinsics_override=intrinsics_override,
    )

    expected_shift = fx_base * (baseline * 0.5) / convergence_depth
    base_cx = float(intrinsics_override[0, 2])
    assert abs(float(left.intrinsics[0, 2]) - (base_cx + expected_shift)) < 2e-5
    assert abs(float(right.intrinsics[0, 2]) - (base_cx - expected_shift)) < 2e-5
