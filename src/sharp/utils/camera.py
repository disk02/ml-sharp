"""Contains utility functionality to render different modalities.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, NamedTuple

import numpy as np
import torch

from .gaussians import Gaussians3D
from .linalg import eyes

TrajetoryType = Literal["swipe", "shake", "rotate", "rotate_forward", "static"]
LookAtMode = Literal["point", "ahead"]


@dataclasses.dataclass
class CameraInfo:
    """Camera info for a pinhole camera."""

    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    width: int
    height: int


class FocusRange(NamedTuple):
    """Parametrizes a range of depth / disparity values."""

    min: float
    focus: float
    max: float


@dataclasses.dataclass
class TrajectoryParams:
    """Parameters for trajectory."""

    type: TrajetoryType = "rotate_forward"
    lookat_mode: LookAtMode = "point"
    max_disparity: float = 0.08
    max_zoom: float = 0.15
    distance_m: float = 0.0
    num_steps: int = 60
    num_repeats: int = 1


def compute_max_offset(
    scene: Gaussians3D,
    params: TrajectoryParams,
    resolution_px: tuple[int, int],
    f_px: float,
) -> np.ndarray:
    """Compute the maximum offset for camera along X/Y/Z axis."""
    scene_points = scene.mean_vectors
    extrinsics = torch.eye(4).to(scene_points.device)
    min_depth, _, _ = _compute_depth_quantiles(scene_points, extrinsics)

    r_px = resolution_px
    diagonal = np.sqrt((r_px[0] / f_px) ** 2 + (r_px[1] / f_px) ** 2)
    max_lateral_offset_m = params.max_disparity * diagonal * min_depth

    max_medial_offset_m = params.max_zoom * min_depth
    max_offset_xyz_m = np.array([max_lateral_offset_m, max_lateral_offset_m, max_medial_offset_m])

    return max_offset_xyz_m


def create_eye_trajectory(
    scene: Gaussians3D,
    params: TrajectoryParams,
    resolution_px: tuple[int, int],
    f_px: float,
) -> list[torch.Tensor]:
    """Create eye trajectory for trajectory type."""
    # Handle static trajectory first to avoid expensive depth computations
    if params.type == "static":
        return create_eye_trajectory_static(
            params.distance_m
        )
    
    # Only compute max_offset for non-static trajectories
    max_offset_xyz_m = compute_max_offset(
        scene,
        params,
        resolution_px,
        f_px,
    )
    # We place the eye trajectory at z=distance plane (default=0),
    # assuming portal plane is placed at z=natural_distance.
    if params.type == "swipe":
        return create_eye_trajectory_swipe(
            max_offset_xyz_m, params.distance_m, params.num_steps, params.num_repeats
        )
    elif params.type == "shake":
        return create_eye_trajectory_shake(
            max_offset_xyz_m, params.distance_m, params.num_steps, params.num_repeats
        )
    elif params.type == "rotate":
        return create_eye_trajectory_rotate(
            max_offset_xyz_m, params.distance_m, params.num_steps, params.num_repeats
        )
    elif params.type == "rotate_forward":
        return create_eye_trajectory_rotate_forward(
            max_offset_xyz_m, params.distance_m, params.num_steps, params.num_repeats
        )
    else:
        raise ValueError(f"Invalid trajectory type {params.type}.")


def create_eye_trajectory_swipe(
    offset_xyz_m: np.ndarray,
    distance_m: float,
    num_steps: int,
    num_repeats: int,
) -> list[torch.Tensor]:
    """Create a left to right swipe trajectory."""
    offset_x_m, _, _ = offset_xyz_m
    eye_positions = [
        torch.tensor([x, 0, distance_m], dtype=torch.float32)
        for x in np.linspace(-offset_x_m, offset_x_m, num_steps)
    ]
    return eye_positions * num_repeats


def create_eye_trajectory_shake(
    offset_xyz_m: np.ndarray,
    distance_m: float,
    num_steps: int,
    num_repeats: int,
) -> list[torch.Tensor]:
    """Create a left right shake followed by an up down shake trajectory."""
    num_steps_total = num_steps * num_repeats
    num_steps_horizontal = num_steps_total // 2
    num_steps_vertical = num_steps_total - num_steps_horizontal

    offset_x_m, offset_y_m, _ = offset_xyz_m
    eye_positions: list[torch.Tensor] = []
    eye_positions.extend(
        torch.tensor(
            [offset_x_m * np.sin(2 * np.pi * t), 0.0, distance_m],
            dtype=torch.float32,
        )
        for t in np.linspace(0, num_repeats, num_steps_horizontal)
    )
    eye_positions.extend(
        torch.tensor(
            [0.0, offset_y_m * np.sin(2 * np.pi * t), distance_m],
            dtype=torch.float32,
        )
        for t in np.linspace(0, num_repeats, num_steps_vertical)
    )

    return eye_positions


def create_eye_trajectory_rotate(
    offset_xyz_m: np.ndarray,
    distance_m: float,
    num_steps: int,
    num_repeats: int,
) -> list[torch.Tensor]:
    """Create a rotating trajectory."""
    num_steps_total = num_steps * num_repeats
    offset_x_m, offset_y_m, _ = offset_xyz_m
    eye_positions = [
        torch.tensor(
            [
                offset_x_m * np.sin(2 * np.pi * t),
                offset_y_m * np.cos(2 * np.pi * t),
                distance_m,
            ],
            dtype=torch.float32,
        )
        for t in np.linspace(0, num_repeats, num_steps_total)
    ]

    return eye_positions


def create_eye_trajectory_rotate_forward(
    offset_xyz_m: np.ndarray,
    distance_m: float,
    num_steps: int,
    num_repeats: int,
) -> list[torch.Tensor]:
    """Create a rotating trajectory."""
    num_steps_total = num_steps * num_repeats
    offset_x_m, _, offset_z_m = offset_xyz_m
    eye_positions = [
        torch.tensor(
            [
                offset_x_m * np.sin(2 * np.pi * t),
                0.0,
                distance_m + offset_z_m * (1.0 - np.cos(2 * np.pi * t)) / 2,
            ],
            dtype=torch.float32,
        )
        for t in np.linspace(0, num_repeats, num_steps_total)
    ]

    return eye_positions


def create_eye_trajectory_static(
    distance_m: float,
) -> list[torch.Tensor]:
    """Create a static trajectory at the canonical center.
    
    Returns a single position at [0, 0, distance_m] which is the canonical center.
    This is used for SBS image rendering where we only need one static view.
    """
    # Single position at the center: [0, 0, distance_m]
    eye_position = torch.tensor([0.0, 0.0, distance_m], dtype=torch.float32)
    # Return a list with just one position
    return [eye_position]


def create_camera_model(
    scene: Gaussians3D,
    intrinsics: torch.Tensor,
    resolution_px: tuple[int, int],
    lookat_mode: LookAtMode = "point",
) -> PinholeCameraModel:
    """Create camera model to simulate general pinhole camera."""
    screen_extrinsics = torch.eye(4)
    screen_intrinsics = intrinsics.clone()

    image_width, image_height = resolution_px
    screen_resolution_px = get_screen_resolution_px_from_input(
        width=image_width, height=image_height
    )

    screen_intrinsics[0] *= screen_resolution_px[0] / image_width
    screen_intrinsics[1] *= screen_resolution_px[1] / image_height

    camera_model = PinholeCameraModel(
        scene,
        screen_extrinsics=screen_extrinsics,
        screen_intrinsics=screen_intrinsics,
        screen_resolution_px=screen_resolution_px,
        focus_depth_quantile=0.1,
        min_depth_focus=2.0,
        lookat_mode=lookat_mode,
    )
    return camera_model


def create_camera_matrix(
    position: torch.Tensor,
    look_at_position: torch.Tensor | None = None,
    world_up: torch.Tensor | None = None,
    inverse: bool = False,
) -> torch.Tensor:
    """Create camera matrix from vectors."""
    device = position.device

    if look_at_position is None:
        look_at_position = torch.zeros(3, device=device)
    if world_up is None:
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)

    position, look_at_position, world_up = torch.broadcast_tensors(
        position, look_at_position, world_up
    )

    camera_front = look_at_position - position
    camera_front = camera_front / camera_front.norm(dim=-1, keepdim=True)

    camera_right = torch.cross(camera_front, world_up, dim=-1)
    camera_right = camera_right / camera_right.norm(dim=-1, keepdim=True)

    camera_down = torch.cross(camera_front, camera_right, dim=-1)
    rotation_matrix = torch.stack([camera_right, camera_down, camera_front], dim=-1)

    matrix = eyes(dim=4, shape=position.shape[:-1], device=device)
    if inverse:
        matrix[..., :3, :3] = rotation_matrix.transpose(-1, -2)
        matrix[..., :3, 3:4] = -rotation_matrix.transpose(-1, -2) @ position[..., None]
    else:
        matrix[..., :3, :3] = rotation_matrix
        matrix[..., :3, 3] = position

    return matrix


def compute_parallel_stereo_pair(
    camera_model: PinholeCameraModel,
    eye_mid: torch.Tensor,
    baseline: float,
    convergence_depth: float | None = None,
    intrinsics_override: torch.Tensor | None = None,
) -> tuple[CameraInfo, CameraInfo]:
    """Compute a parallel stereo pair with off-axis convergence via principal point shift."""
    device = eye_mid.device
    dtype = eye_mid.dtype

    if intrinsics_override is None:
        base_intrinsics = camera_model.screen_intrinsics.to(device=device, dtype=dtype).clone()
    else:
        base_intrinsics = intrinsics_override.to(device=device, dtype=dtype).clone()
    screen_extrinsics = camera_model.screen_extrinsics.to(device=device, dtype=dtype)

    origin = (
        eye_mid
        if camera_model.lookat_mode == "ahead"
        else torch.zeros(3, device=device, dtype=dtype)
    )

    if camera_model.lookat_point is None:
        depth_focus = max(camera_model.min_depth_focus, camera_model.depth_quantiles.focus)
        look_at_position = origin + torch.tensor([0.0, 0.0, depth_focus], device=device, dtype=dtype)
    else:
        look_at_position = origin + torch.tensor(
            [*camera_model.lookat_point], device=device, dtype=dtype
        )

    world_up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)

    camera_front = look_at_position - eye_mid
    front_norm = camera_front.norm()
    if front_norm < 1e-6:
        camera_front = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        camera_front = camera_front / front_norm

    camera_right = torch.cross(camera_front, world_up, dim=-1)
    right_norm = camera_right.norm()
    if right_norm < 1e-6:
        camera_right = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    else:
        camera_right = camera_right / right_norm

    camera_down = torch.cross(camera_front, camera_right, dim=-1)
    camera_down = camera_down / camera_down.norm()
    camera_front = torch.cross(camera_right, camera_down, dim=-1)
    camera_front = camera_front / camera_front.norm()
    rotation_matrix = torch.stack([camera_right, camera_down, camera_front], dim=-1)

    half_baseline = float(baseline) * 0.5
    eye_position_l = eye_mid - camera_right * half_baseline
    eye_position_r = eye_mid + camera_right * half_baseline

    rotation_wc = rotation_matrix.transpose(-1, -2)
    extrinsics_l = torch.eye(4, device=device, dtype=dtype)
    extrinsics_r = torch.eye(4, device=device, dtype=dtype)
    extrinsics_l[:3, :3] = rotation_wc
    extrinsics_r[:3, :3] = rotation_wc
    extrinsics_l[:3, 3] = -(rotation_wc @ eye_position_l)
    extrinsics_r[:3, 3] = -(rotation_wc @ eye_position_r)
    extrinsics_l = extrinsics_l @ screen_extrinsics
    extrinsics_r = extrinsics_r @ screen_extrinsics

    focus_depth = max(camera_model.min_depth_focus, camera_model.depth_quantiles.focus)
    convergence_z = convergence_depth if convergence_depth is not None else focus_depth
    if convergence_z <= 0:
        convergence_z = focus_depth

    fx = float(base_intrinsics[0, 0])
    cx_shift = fx * half_baseline / convergence_z if convergence_z > 0 else 0.0
    intrinsics_l = base_intrinsics.clone()
    intrinsics_r = base_intrinsics.clone()
    intrinsics_l[0, 2] = intrinsics_l[0, 2] + cx_shift
    intrinsics_r[0, 2] = intrinsics_r[0, 2] - cx_shift

    return (
        CameraInfo(
            intrinsics=intrinsics_l,
            extrinsics=extrinsics_l,
            width=camera_model.screen_resolution_px[0],
            height=camera_model.screen_resolution_px[1],
        ),
        CameraInfo(
            intrinsics=intrinsics_r,
            extrinsics=extrinsics_r,
            width=camera_model.screen_resolution_px[0],
            height=camera_model.screen_resolution_px[1],
        ),
    )


def make_cylindrical_rays(
    width: int,
    height: int,
    hfov_deg: float,
    fy: float,
    cy: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build normalized camera-space rays for cylindrical projection.

    Horizontal coordinates map to yaw angle theta over ``hfov_deg`` while
    vertical coordinates preserve pinhole mapping from ``fy`` and ``cy``.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    hfov_rad = torch.tensor(hfov_deg * np.pi / 180.0, device=device, dtype=dtype)
    if hfov_rad <= 0:
        raise ValueError("hfov_deg must be > 0")

    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    # theta=0 is always at the screen center. Per-eye principal-point shifts
    # are applied in intrinsics and handle off-axis stereo convergence.
    theta = ((uu + 0.5) / width - 0.5) * hfov_rad
    x_img = torch.tan(theta)
    y_img = ((vv + 0.5) - cy) / fy

    rays = torch.stack([x_img, y_img, torch.ones_like(x_img)], dim=-1)
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return rays


class PinholeCameraModel:
    """Camera model that focuses on point."""

    def __init__(
        self,
        scene: Gaussians3D,
        screen_extrinsics: torch.Tensor,
        screen_intrinsics: torch.Tensor,
        screen_resolution_px: tuple[int, int],
        focus_depth_quantile: float = 0.1,
        min_depth_focus: float = 2.0,
        lookat_point: tuple[float, float, float] | None = None,
        lookat_mode: LookAtMode = "point",
    ) -> None:
        """Initialize GeneralPinholeCameraModel.

        Args:
            scene: The scene to display.
            screen_extrinsics: Extrinsics of the default position.
            screen_intrinsics: Intrinsics to use for rendering.
            screen_resolution_px: Width and height to render.
            focus_depth_quantile: Where inside the depth range to focus on.
            min_depth_focus: Depth to focus at.
            lookat_point: a point that the camera's Z axis directs towards.
            lookat_mode: "point" to look at a fixed point,
                "ahead" to look straight ahead.
        """
        self.scene = scene
        self.screen_extrinsics = screen_extrinsics
        self.screen_intrinsics = screen_intrinsics
        self.screen_resolution_px = screen_resolution_px

        self.focus_depth_quantile = focus_depth_quantile
        self.min_depth_focus = min_depth_focus
        self.lookat_point = lookat_point
        self.lookat_mode = lookat_mode

        scene_points = scene.mean_vectors
        if scene_points.ndim == 3:
            scene_points = scene_points[0]
        elif scene_points.ndim != 2:
            raise ValueError("Unsupported dimensionality of scene points.")
        self._scene_points = scene_points.cpu()

        self.depth_quantiles = _compute_depth_quantiles(
            self._scene_points,
            self.screen_extrinsics,
            q_focus=self.focus_depth_quantile,
        )

    def compute(self, eye_pos: torch.Tensor) -> CameraInfo:
        """Compute camera for eye position."""
        extrinsics = self.screen_extrinsics.clone()

        origin = eye_pos if self.lookat_mode == "ahead" else torch.zeros(3)

        if self.lookat_point is None:
            depth_focus = max(self.min_depth_focus, self.depth_quantiles.focus)
            look_at_position = origin + torch.tensor([0.0, 0.0, depth_focus])
        else:
            look_at_position = origin + torch.tensor([*self.lookat_point])

        world_up = torch.tensor([0.0, -1.0, 0.0])
        extrinsics_modifier = create_camera_matrix(
            eye_pos, look_at_position, world_up, inverse=True
        )
        extrinsics = extrinsics_modifier @ self.screen_extrinsics

        camera_info = CameraInfo(
            intrinsics=self.screen_intrinsics,
            extrinsics=extrinsics,
            width=self.screen_resolution_px[0],
            height=self.screen_resolution_px[1],
        )
        return camera_info

    def set_screen_extrinsics(self, new_value: torch.Tensor) -> None:
        """Modify the default extrinsics."""
        self.screen_extrinsics = new_value
        self.depth_quantiles = _compute_depth_quantiles(self._scene_points, self.screen_extrinsics)


def get_screen_resolution_px_from_input(width: int, height: int) -> tuple[int, int]:
    """Get resolution for metadata dictionary."""
    resolution_px = (width, height)
    # halve the dimensions for super large image
    if resolution_px[1] > 3000:
        resolution_px = (resolution_px[0] // 2, resolution_px[1] // 2)
    # for mp4 compatibility, enforce dimensions to even number,
    # otherwise could not be played in browser
    if resolution_px[0] % 2 != 0:
        resolution_px = (resolution_px[0] + 1, resolution_px[1])
    if resolution_px[1] % 2 != 0:
        resolution_px = (resolution_px[0], resolution_px[1] + 1)
    return resolution_px


def _compute_depth_quantiles(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    q_near: float = 0.001,
    q_focus: float = 0.1,
    q_far: float = 0.999,
) -> FocusRange:
    """Compute disparity quantiles for scene and extrinsics id."""
    points_local = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    depth_values = points_local[..., 2].flatten()
    depth_values = depth_values[depth_values > 0]
    q_values = torch.tensor([q_near, q_focus, q_far])
    depth_quantiles_pt = torch.quantile(depth_values.cpu(), q_values)
    depth_quantiles = FocusRange(
        min=float(depth_quantiles_pt[0]),
        focus=float(depth_quantiles_pt[1]),
        max=float(depth_quantiles_pt[2]),
    )
    return depth_quantiles
