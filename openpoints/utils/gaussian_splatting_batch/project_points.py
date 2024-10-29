import torch
from einops import repeat


def project_points(
        xyz: torch.Tensor,
        intr: torch.Tensor,
        extr: torch.Tensor,
        W: int,
        H: int,
        nearest: float = 0.2,
        extent: float = 1.3,
):
    """Project 3D points to the screen.
        a pytorch implementation support batch @msplat

    Args:
        xyz (torch.Tensor): points, [B, N, 3] or [N, 3]
        intr (torch.Tensor): camera intrinsics, [B, 4]
        extr (torch.Tensor): camera pose, [B, 4, 4]
        W (int): view width of camera field size.
        H (int): view height of camera field size.
        nearest (float, optional): nearest threshold for frustum culling, by default 0.2
        extent (float, optional): extent threshold for frustum culling, by default 1.3

    Returns
        uv (torch.Tensor): 2D positions for each point in the image, [B, N, 2]
        depth (torch.Tensor): depth for each point, [B, N, 1]
    """
    assert len(intr.shape) == 2
    B = intr.shape[0]
    device = xyz.device

    K = torch.eye(3, device=device)
    K = repeat(K, 'm d -> b m d', b=B)
    K = K.clone()
    K[:, 0, 0] = intr[:, 0]
    K[:, 1, 1] = intr[:, 1]
    K[:, 0, 2] = intr[:, 2]
    K[:, 1, 2] = intr[:, 3]
    R = extr[:, :3, :3]
    t = extr[:, :3, -1].unsqueeze(dim=-1)

    xyz_t = xyz.permute(0, 2, 1) if len(xyz.shape) == 3 else xyz.permute(1, 0)
    pt_cam = torch.matmul(R, xyz_t)
    pt_cam = pt_cam + t
    # Apply camera intrinsic matrix
    p_proj = torch.matmul(K[:, :], pt_cam)

    depths = repeat(p_proj[:, 2], 'b m -> b d m', d=2)
    uv = p_proj[:, :2] / depths - 0.5
    uv = uv.permute(0, 2, 1)

    depths = torch.nan_to_num(depths[:, 0]).squeeze(-1)
    near_mask = depths <= nearest
    extent_mask_x = torch.logical_or(uv[:, :, 0] < (1 - extent) * W * 0.5,
                                     uv[:, :, 0] > (1 + extent) * W * 0.5)
    extent_mask_y = torch.logical_or(uv[:, :, 1] < (1 - extent) * H * 0.5,
                                     uv[:, :, 1] > (1 + extent) * H * 0.5)
    extent_mask = torch.logical_or(extent_mask_x, extent_mask_y)
    mask = torch.logical_or(near_mask, extent_mask)
    uv_masked = uv.clone()
    depths_masked = depths.clone()
    uv_masked[:, :, 0][mask] = 0
    uv_masked[:, :, 1][mask] = 0
    depths_masked[mask] = 0

    return uv_masked, depths_masked.unsqueeze(-1)


if __name__ == '__main__':
    xyz = torch.randn((1, 10, 3), device='cpu')
    intr = torch.randn((1, 4,), device='cpu')
    extr = torch.randn((1, 4, 4), device='cpu')
    uv, depths = project_points(xyz, intr, extr, 512, 512)
    print(uv.shape, depths.shape)
