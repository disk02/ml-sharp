import torch

from sharp.utils.gaussians import Gaussians3D, prune_gaussians


def _make_gaussians(opacities: torch.Tensor, scales: torch.Tensor) -> Gaussians3D:
    num = opacities.shape[0]
    mean_vectors = torch.arange(float(num)).unsqueeze(1).repeat(1, 3)
    quaternions = torch.zeros((num, 4))
    colors = torch.zeros((num, 3))
    singular_values = scales
    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def test_prune_thresholds() -> None:
    opacities = torch.tensor([0.1, 0.6, 0.2, 0.9, 0.4])
    scales = torch.tensor(
        [
            [0.05, 0.05, 0.05],
            [0.1, 0.2, 0.1],
            [0.3, 0.1, 0.2],
            [0.5, 0.4, 0.2],
            [0.15, 0.05, 0.1],
        ]
    )
    gaussians = _make_gaussians(opacities, scales)
    pruned = prune_gaussians(gaussians, min_opacity=0.5, min_scale=0.15)
    assert pruned.mean_vectors[:, 0].tolist() == [1.0, 3.0]


def test_prune_topk_by_score() -> None:
    opacities = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3])
    scales = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.5, 0.1, 0.2],
            [0.2, 0.2, 0.2],
            [0.4, 0.1, 0.1],
        ]
    )
    gaussians = _make_gaussians(opacities, scales)
    pruned = prune_gaussians(gaussians, max_splats=2, score="opacity_scale")
    assert pruned.mean_vectors[:, 0].tolist() == [3.0, 4.0]


def test_prune_fallback_keeps_one() -> None:
    opacities = torch.tensor([0.1, 0.2, 0.3])
    scales = torch.tensor([[0.05, 0.05, 0.05]]).repeat(3, 1)
    gaussians = _make_gaussians(opacities, scales)
    pruned = prune_gaussians(gaussians, min_opacity=0.9, min_scale=0.5)
    assert pruned.mean_vectors.shape[0] == 1
