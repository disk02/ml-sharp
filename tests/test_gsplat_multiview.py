import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for gsplat tests")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for gsplat multiview test", allow_module_level=True)

from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer


def test_gsplat_multiview_batch_rendering() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_points = 512

    mean_vectors = torch.randn(1, num_points, 3, device=device, dtype=torch.float32) * 0.1
    mean_vectors[..., 2] = torch.rand(1, num_points, device=device) * 0.5 + 2.0

    quaternions = torch.randn(1, num_points, 4, device=device, dtype=torch.float32)
    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)

    singular_values = torch.rand(1, num_points, 3, device=device, dtype=torch.float32) * 0.2 + 0.05
    colors = torch.rand(1, num_points, 3, device=device, dtype=torch.float32)
    opacities = torch.rand(1, num_points, 1, device=device, dtype=torch.float32)

    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        quaternions=quaternions,
        singular_values=singular_values,
        colors=colors,
        opacities=opacities,
    )

    extrinsics = torch.eye(4, device=device, dtype=torch.float32).repeat(2, 1, 1)
    extrinsics[1, 0, 3] = 0.1

    intrinsics = torch.eye(4, device=device, dtype=torch.float32).repeat(2, 1, 1)
    intrinsics[:, 0, 0] = 50.0
    intrinsics[:, 1, 1] = 50.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0

    renderer = GSplatRenderer(color_space="linearRGB")
    rendering = renderer(
        gaussians=gaussians,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_width=64,
        image_height=64,
    )

    assert rendering.color.shape[0] == 2, (
        "GSplatRenderer did not return batched outputs for 2 views; "
        "multi-view batching is not supported by the current gsplat path."
    )
    assert rendering.color.shape[1] == 3
    assert rendering.color.shape[-2:] == (64, 64)
    assert rendering.depth.shape[0] == 2
    assert rendering.depth.shape[-2:] == (64, 64)
    assert rendering.alpha.shape[0] == 2
    assert rendering.alpha.shape[-2:] == (64, 64)
