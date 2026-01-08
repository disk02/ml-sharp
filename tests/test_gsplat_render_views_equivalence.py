import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for gsplat tests")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for gsplat tests", allow_module_level=True)

from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer


@pytest.mark.parametrize("num_gaussians", [512])
def test_render_views_matches_legacy_forward(num_gaussians: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    mean_vectors = torch.empty((1, num_gaussians, 3), device=device, dtype=dtype).uniform_(
        -0.5, 0.5
    )
    mean_vectors[..., 2] = torch.empty(
        (1, num_gaussians), device=device, dtype=dtype
    ).uniform_(2.0, 2.5)

    quaternions = torch.randn((1, num_gaussians, 4), device=device, dtype=dtype)
    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)

    singular_values = torch.empty(
        (1, num_gaussians, 3), device=device, dtype=dtype
    ).uniform_(0.01, 0.05)

    colors = torch.rand((1, num_gaussians, 3), device=device, dtype=dtype)
    opacities = torch.rand((1, num_gaussians), device=device, dtype=dtype)

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )

    extrinsics = torch.eye(4, device=device, dtype=dtype)
    intrinsics = torch.eye(4, device=device, dtype=dtype)
    intrinsics[0, 0] = 50.0
    intrinsics[1, 1] = 50.0
    intrinsics[0, 2] = 32.0
    intrinsics[1, 2] = 32.0

    renderer = GSplatRenderer(color_space="linearRGB")

    legacy = renderer.forward(
        gaussians=gaussians,
        extrinsics=extrinsics[None],
        intrinsics=intrinsics[None],
        image_width=64,
        image_height=64,
    )
    new = renderer.render_views(
        gaussians=gaussians,
        extrinsics=extrinsics[None],
        intrinsics=intrinsics[None],
        image_width=64,
        image_height=64,
    )

    legacy_color = legacy.color
    new_color = new.color

    if legacy_color.dtype == torch.uint8 and new_color.dtype == torch.uint8:
        diff = (legacy_color.to(torch.int32) - new_color.to(torch.int32)).abs()
        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.float().mean().item()
        max_threshold = 2
        mean_threshold = 0.1
    else:
        legacy_float = legacy_color
        new_float = new_color
        if legacy_float.dtype == torch.uint8:
            legacy_float = legacy_float.float() / 255.0
        else:
            legacy_float = legacy_float.float()
        if new_float.dtype == torch.uint8:
            new_float = new_float.float() / 255.0
        else:
            new_float = new_float.float()
        diff = (legacy_float - new_float).abs()
        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.mean().item()
        max_threshold = 1e-3
        mean_threshold = 1e-4

    assert max_abs_diff <= max_threshold and mean_abs_diff <= mean_threshold, (
        "GSplatRenderer render mismatch for C=1. "
        f"legacy dtype/shape={legacy_color.dtype}/{tuple(legacy_color.shape)}, "
        f"new dtype/shape={new_color.dtype}/{tuple(new_color.shape)}, "
        f"max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}"
    )
