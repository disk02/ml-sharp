import torch

from sharp.utils.gaussians import Gaussians3D, build_camera_stats_scene


def _make_scene(num_points: int, batched: bool = False) -> Gaussians3D:
    x = torch.arange(num_points, dtype=torch.float32)
    scene = Gaussians3D(
        mean_vectors=torch.stack([x, x * 0.5, x * -1.0], dim=1),
        singular_values=torch.ones((num_points, 3), dtype=torch.float32),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_points, dtype=torch.float32),
        colors=torch.ones((num_points, 3), dtype=torch.float32),
        opacities=torch.ones((num_points,), dtype=torch.float32),
    )
    if batched:
        return Gaussians3D(
            mean_vectors=scene.mean_vectors.unsqueeze(0),
            singular_values=scene.singular_values.unsqueeze(0),
            quaternions=scene.quaternions.unsqueeze(0),
            colors=scene.colors.unsqueeze(0),
            opacities=scene.opacities.unsqueeze(0),
        )
    return scene


def _unprojection() -> torch.Tensor:
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[0, 0] = 2.0
    matrix[1, 1] = 3.0
    matrix[2, 2] = 4.0
    matrix[0, 3] = 10.0
    matrix[1, 3] = -5.0
    matrix[2, 3] = 7.5
    return matrix


def test_build_camera_stats_scene_is_deterministic() -> None:
    scene = _make_scene(100_000)
    unprojection = _unprojection()
    torch.manual_seed(0)
    first = build_camera_stats_scene(scene, unprojection)
    torch.manual_seed(987_654_321)
    second = build_camera_stats_scene(scene, unprojection)
    for field in ("mean_vectors", "singular_values", "quaternions", "colors", "opacities"):
        assert torch.equal(getattr(first, field), getattr(second, field))


def test_build_camera_stats_scene_stride_is_uniform() -> None:
    num_points = 100_000
    scene = _make_scene(num_points)
    result = build_camera_stats_scene(scene, torch.eye(4, dtype=torch.float32))
    count = result.mean_vectors.shape[0]
    assert count <= 16_384
    # X coordinates are arange(num_points), so the sampled stride is directly observable.
    x = result.mean_vectors[:, 0]
    assert x[0] == 0.0
    diffs = torch.diff(x)
    assert torch.all(diffs == diffs[0]) and diffs[0] > 0
    # The stride must span the whole array, not just the leading part of it.
    assert x[-1] > 0.95 * num_points


def test_build_camera_stats_scene_applies_world_transform() -> None:
    num_points = 100
    scene = _make_scene(num_points)
    unprojection = _unprojection()
    result = build_camera_stats_scene(scene, unprojection)
    expected = scene.mean_vectors @ unprojection[:3, :3].T + unprojection[:3, 3]
    assert result.mean_vectors.shape[0] == num_points
    assert torch.allclose(result.mean_vectors, expected, atol=1e-4)


def test_build_camera_stats_scene_batched_input_matches_unbatched() -> None:
    scene = _make_scene(4000, batched=True)
    unprojection = _unprojection()
    batched_result = build_camera_stats_scene(scene, unprojection)
    unbatched_scene = Gaussians3D(
        mean_vectors=scene.mean_vectors[0],
        singular_values=scene.singular_values[0],
        quaternions=scene.quaternions[0],
        colors=scene.colors[0],
        opacities=scene.opacities[0],
    )
    unbatched_result = build_camera_stats_scene(unbatched_scene, unprojection)
    for field in ("mean_vectors", "singular_values", "quaternions", "colors", "opacities"):
        assert torch.equal(getattr(batched_result, field), getattr(unbatched_result, field))
