import torch
from einops import repeat


def compute_cov3d(xyz_padded: torch.Tensor, visible: torch.Tensor = None) -> torch.Tensor:
    """Computes the per-point covariance matrices by of the 3D locations of each points by sampled points.
        a pytorch implementation support batch @pytorch3d

    Args:
        xyz_padded (torch.Tensor): padded points by neighbors, [B, N, n_neighbors, 3]
        visible (torch.Tensor): a visible mask, [B, N, 1, *]
    Returns
        cov3d (torch.Tensor): the 3D covariance matrices, [B, N, 3, 3]
    """
    B, N, _, _ = xyz_padded.shape
    if visible is None:
        visible = torch.ones((B, N, 1), device=xyz_padded.device)
    else:
        visible = visible.sum(dim=-1, keepdim=False)
        visible = (visible > 0).int()
    visible = visible.squeeze(dim=-1)
    visible = repeat(visible, 'b n -> b n p q', p=3, q=3)

    # obtain the mean of the neighborhood
    pt_mean = xyz_padded.mean(2, keepdim=True)
    # compute the diff of the neighborhood and the mean of the neighborhood
    central_diff = xyz_padded - pt_mean
    # per-nn-point covariances
    cov3d = central_diff.unsqueeze(4) * central_diff.unsqueeze(3)
    # per-point covariances
    cov3d = cov3d.mean(2)
    return cov3d.mul(visible)


def compute_scaling_rotation(cov3d: torch.Tensor):
    """
    :param cov3d: [B, N, 3, 3]
    :return scaling: [B, N, 3]
    :return rotation: [B, N, 3, 3]
    """
    U, S, _ = torch.linalg.svd(cov3d)
    return torch.sqrt(S), U


if __name__ == '__main__':
    import time
    xyz_sampled = torch.randn((8, 6000, 32, 3))

    cov3d = compute_cov3d(xyz_sampled)
    start_time = time.time()
    cov3d = compute_cov3d(xyz_sampled)
    end_time = time.time()
    print(end_time - start_time)
    print(cov3d.shape)

    L, Q = torch.linalg.eigh(cov3d)
    print(L.shape, Q.shape)

    scaling, rotation = compute_scaling_rotation(cov3d)
    start_time = time.time()
    scaling, rotation = compute_scaling_rotation(cov3d)
    end_time = time.time()
    print(end_time - start_time)
    print(scaling.shape, rotation.shape)

